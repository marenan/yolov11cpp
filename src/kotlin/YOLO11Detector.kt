package com.example.opencv_tutorial

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
//import android.util.Log

/**
 * YOLOv11Detector for Android using TFLite and OpenCV
 *
 * This class handles object detection using the YOLOv11 model with TensorFlow Lite
 * for inference and OpenCV for image processing.
 */
class YOLO11Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelsPath: String,
    useGPU: Boolean = true
) {
    // Detection parameters - matching C++ implementation
    companion object {
        // Match the C++ implementation thresholds
        const val CONFIDENCE_THRESHOLD = 0.25f  // Changed from 0.4f to match C++ code
        const val IOU_THRESHOLD = 0.45f         // Changed from 0.3f to match C++ code
        private const val TAG = "YOLO11Detector"
    }

    // Data structures for model and inference
    private var interpreter: Interpreter
    private val classNames: List<String>
    private val classColors: List<IntArray>
    private var gpuDelegate: GpuDelegate? = null

    // Input shape info
    private var inputWidth: Int = 640
    private var inputHeight: Int = 640
    private var isQuantized: Boolean = false
    private var numClasses: Int = 0

    init {
        // Load model with proper options
        val tfliteOptions = Interpreter.Options()

        // GPU Delegate setup if available and requested
        if (useGPU) {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                debug("GPU acceleration enabled")
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
                tfliteOptions.addDelegate(gpuDelegate)
            } else {
                debug("GPU acceleration not supported, using CPU")
                tfliteOptions.setNumThreads(4)
            }
        } else {
            debug("Using CPU for inference")
            tfliteOptions.setNumThreads(4)
        }

        // Load the TFLite model
        val modelBuffer = loadModelFile(modelPath)
        interpreter = Interpreter(modelBuffer, tfliteOptions)

        // Get input shape information
        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        isQuantized = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8

        // Get output shape information to determine number of classes
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        numClasses = outputShape[1] - 4 // The output contains [x, y, w, h, class1, class2, ...]

        debug("Model loaded with input dimensions: $inputWidth x $inputHeight")
        debug("Model uses ${if(isQuantized) "quantized" else "float"} input")
        debug("Model output shape: ${outputShape.joinToString()}")
        debug("Detected $numClasses classes")

        // Load class names and generate colors
        classNames = loadClassNames(labelsPath)
        classColors = generateColors(classNames.size)

        if (classNames.size != numClasses) {
            debug("Warning: Number of classes in label file (${classNames.size}) differs from model output ($numClasses)")
        }

        debug("Loaded ${classNames.size} classes")
    }

    /**
     * Main detection function that processes an image and returns detected objects
     */
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = CONFIDENCE_THRESHOLD,
               iouThreshold: Float = IOU_THRESHOLD): List<Detection> {
        val startTime = SystemClock.elapsedRealtime()
        debug("Starting detection with conf=$confidenceThreshold, iou=$iouThreshold")

        // Convert Bitmap to Mat for OpenCV processing
        val inputMat = Mat()
        Utils.bitmapToMat(bitmap, inputMat)
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2BGR)

        // Prepare input for TFLite
        val originalSize = Size(bitmap.width.toDouble(), bitmap.height.toDouble())
        val resizedImgMat = Mat() // Will hold the resized image

        // First preprocess using OpenCV (exactly like C++ version)
        val inputTensor = preprocessImageOpenCV(
            inputMat,
            resizedImgMat,
            Size(inputWidth.toDouble(), inputHeight.toDouble())
        )

        // Run inference
        val outputs = runInference(inputTensor)

        // Process outputs to get detections
        val detections = postprocess(
            outputs,
            originalSize,
            Size(inputWidth.toDouble(), inputHeight.toDouble()),
            confidenceThreshold,
            iouThreshold
        )

        // Clean up
        inputMat.release()
        resizedImgMat.release()

        val inferenceTime = SystemClock.elapsedRealtime() - startTime
        debug("Detection completed in $inferenceTime ms with ${detections.size} objects")

        return detections
    }

    /**
     * Preprocess the input image using OpenCV to match the C++ implementation exactly
     */
    private fun preprocessImageOpenCV(image: Mat, outImage: Mat, newShape: Size): ByteBuffer {
        val scopedTimer = ScopedTimer("preprocessing")

        // Resize with letterboxing to maintain aspect ratio (identical to C++ version)
        letterBox(image, outImage, newShape, Scalar(114.0, 114.0, 114.0))

        // Convert BGR to RGB (YOLOv11 expects RGB input)
        val rgbMat = Mat()
        Imgproc.cvtColor(outImage, rgbMat, Imgproc.COLOR_BGR2RGB)

        // DEBUG: Output dimensions for verification
        debug("Preprocessed image dimensions: ${rgbMat.width()}x${rgbMat.height()}")

        // Prepare the ByteBuffer to store the model input data
        val bytesPerChannel = if (isQuantized) 1 else 4
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputWidth * inputHeight * 3 * bytesPerChannel)
        inputBuffer.order(ByteOrder.nativeOrder())

        try {
            // Convert to proper format for TFLite
            if (isQuantized) {
                // For quantized models, prepare as bytes
                val pixels = ByteArray(rgbMat.width() * rgbMat.height() * rgbMat.channels())
                rgbMat.get(0, 0, pixels)

                for (i in pixels.indices) {
                    inputBuffer.put(pixels[i])
                }
            } else {
                // For float models, normalize to [0,1]
                // CRITICAL: Create a normalized float Mat directly using OpenCV for better precision
                val normalizedMat = Mat()
                rgbMat.convertTo(normalizedMat, CvType.CV_32FC3, 1.0/255.0)

                // Now copy the normalized float values to TFLite input buffer
                val floatValues = FloatArray(normalizedMat.width() * normalizedMat.height() * normalizedMat.channels())
                normalizedMat.get(0, 0, floatValues)

                for (value in floatValues) {
                    inputBuffer.putFloat(value)
                }

                normalizedMat.release()
            }
        } catch (e: Exception) {
            debug("Error during preprocessing: ${e.message}")
            e.printStackTrace()
        }

        inputBuffer.rewind()
        rgbMat.release()

        scopedTimer.stop()
        return inputBuffer
    }

    /**
     * Runs inference with TFLite and returns the raw output
     */
    private fun runInference(inputBuffer: ByteBuffer): Map<Int, Any> {
        val scopedTimer = ScopedTimer("inference")

        val outputs: MutableMap<Int, Any> = HashMap()

        try {
            // YOLOv11 with TFLite typically outputs a single tensor
            val outputShape = interpreter.getOutputTensor(0).shape()
            debug("Output tensor shape: ${outputShape.joinToString()}")

            // Correctly allocate output buffer based on the shape
            if (isQuantized) {
                val outputSize = outputShape.reduce { acc, i -> acc * i }
                val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize)
                    .order(ByteOrder.nativeOrder())
                outputs[0] = outputBuffer

                // Run inference with quantized model
                interpreter.run(inputBuffer, outputBuffer)
            } else {
                val outputSize = outputShape.reduce { acc, i -> acc * i }
                val outputBuffer = ByteBuffer.allocateDirect(4 * outputSize)
                    .order(ByteOrder.nativeOrder())
                outputs[0] = outputBuffer

                // Run inference with float model
                interpreter.run(inputBuffer, outputBuffer)

                // Debug: Peek at some values to verify output format
                outputBuffer.rewind()
                val values = FloatArray(min(10, outputSize))
                for (i in values.indices) {
                    values[i] = outputBuffer.float
                }
                debug("First few output values: ${values.joinToString()}")
                outputBuffer.rewind()
            }
        } catch (e: Exception) {
            debug("Error during inference: ${e.message}")
            e.printStackTrace()
        }

        scopedTimer.stop()
        return outputs
    }

    /**
     * Post-processes the model outputs to extract detections in the exact same way as C++
     */
    private fun postprocess(
        outputMap: Map<Int, Any>,
        originalImageSize: Size,
        resizedImageShape: Size,
        confThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        val scopedTimer = ScopedTimer("postprocessing")

        val detections = mutableListOf<Detection>()

        try {
            // Get output buffer
            val outputBuffer = outputMap[0] as ByteBuffer
            outputBuffer.rewind()

            // Get output dimensions - match the C++ version's expectations
            val outputShapes = interpreter.getOutputTensor(0).shape()
            debug("Output tensor shape: ${outputShapes.joinToString()}")

            // Using correct indexing based on tensor shape: [batch, features, detections]
            val num_features = outputShapes[1]  // Total features = 4 box coords + num classes
            val num_detections = outputShapes[2] // Number of detection boxes

            // Calculate number of classes (features minus 4 coordinates)
            numClasses = num_features - 4

            debug("Processing output tensor: features=$num_features, detections=$num_detections, classes=$numClasses")

            // Extract boxes, confidences, and class ids
            val boxes = mutableListOf<RectF>()
            val confidences = mutableListOf<Float>()
            val classIds = mutableListOf<Int>()
            val nmsBoxes = mutableListOf<RectF>() // For class-separated NMS

            // Create a float array from the buffer for more efficient access
            val outputArray = FloatArray(outputShapes[0] * num_features * num_detections)
            outputBuffer.rewind()
            for (i in outputArray.indices) {
                outputArray[i] = outputBuffer.float
            }

            // Process each detection with corrected access pattern
            for (d in 0 until num_detections) {
                // Read coordinates - fix the access pattern to match C++ implementation
                // The data is arranged as [x1,x2,...,xn, y1,y2,...,yn, w1,w2,...,wn, h1,h2,...,hn, scores...]
                val centerX = outputArray[0 * num_detections + d]
                val centerY = outputArray[1 * num_detections + d]
                val width = outputArray[2 * num_detections + d]
                val height = outputArray[3 * num_detections + d]

                // Find class with maximum score
                var maxScore = -Float.MAX_VALUE
                var classId = -1

                // Read class scores with corrected access pattern
                for (c in 0 until numClasses) {
                    val score = outputArray[(4 + c) * num_detections + d]
                    if (score > maxScore) {
                        maxScore = score
                        classId = c
                    }
                }

                // Keep detections above threshold
                if (maxScore > confThreshold) {
                    // Convert center coordinates to top-left format
                    val left = centerX - width / 2.0f
                    val top = centerY - height / 2.0f
                    val right = centerX + width / 2.0f
                    val bottom = centerY + height / 2.0f

                    // Log coordinates for debugging
                    debug("Detection $d: center=($centerX,$centerY), wh=($width,$height), score=$maxScore, class=$classId")

                    // Scale coordinates to original image size
                    val scaledBox = scaleCoords(
                        resizedImageShape,
                        RectF(left, top, right, bottom),
                        originalImageSize
                    )

                    // Validate the box has positive dimensions before adding
                    val boxWidth = scaledBox.right - scaledBox.left
                    val boxHeight = scaledBox.bottom - scaledBox.top
                    
                    if (boxWidth > 1 && boxHeight > 1) {  // Require at least 1x1 pixel
                        // Round coordinates to integer precision (match C++ implementation)
                        val roundedBox = RectF(
                            round(scaledBox.left),
                            round(scaledBox.top),
                            round(scaledBox.right),
                            round(scaledBox.bottom)
                        )

                        // Create offset box for NMS with class separation (exactly like C++)
                        val nmsBox = RectF(
                            roundedBox.left + classId * 7680f,
                            roundedBox.top + classId * 7680f,
                            roundedBox.right + classId * 7680f,
                            roundedBox.bottom + classId * 7680f
                        )

                        nmsBoxes.add(nmsBox)
                        boxes.add(roundedBox)
                        confidences.add(maxScore)
                        classIds.add(classId)
                    } else {
                        debug("Skipped detection with invalid dimensions: ${boxWidth}x${boxHeight}")
                    }
                }
            }

            debug("Found ${boxes.size} raw detections before NMS")

            // Run NMS to eliminate redundant boxes
            val selectedIndices = mutableListOf<Int>()
            nonMaxSuppression(nmsBoxes, confidences, confThreshold, iouThreshold, selectedIndices)

            debug("After NMS: ${selectedIndices.size} detections remaining")

            // Create final detection objects
            for (idx in selectedIndices) {
                val box = boxes[idx]

                // Fix: Calculate width and height correctly from left, top, right, bottom
                val width = box.right - box.left
                val height = box.bottom - box.top

                // Skip invalid boxes
                if (width <= 0 || height <= 0) {
                    debug("Skipped invalid box with dimensions: ${width}x${height}")
                    continue
                }

                // Verify class ID is valid
                val validClassId = if (classIds[idx] >= 0 && classIds[idx] < classNames.size) {
                    classIds[idx]
                } else {
                    debug("Invalid class ID: ${classIds[idx]}. Using fallback.")
                    0 // Fallback to first class
                }

                // Create detection with correct coordinates
                val detection = Detection(
                    BoundingBox(
                        box.left.toInt(),
                        box.top.toInt(),
                        width.toInt(),
                        height.toInt()
                    ),
                    confidences[idx],
                    validClassId
                )

                detections.add(detection)
                debug("Added detection: box=${detection.box.x},${detection.box.y},${detection.box.width},${detection.box.height}, " +
                      "conf=${detection.conf}, class=${validClassId}")
            }
        } catch (e: Exception) {
            debug("Error during postprocessing: ${e.message}")
            e.printStackTrace()
        }

        scopedTimer.stop()
        return detections
    }

    /**
     * Draws bounding boxes on the provided bitmap
     */
    fun drawDetections(bitmap: Bitmap, detections: List<Detection>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = max(bitmap.width, bitmap.height) * 0.004f

        val textPaint = Paint()
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = max(bitmap.width, bitmap.height) * 0.02f

        // Filter detections to ensure quality results
        val filteredDetections = detections.filter {
            it.conf > CONFIDENCE_THRESHOLD &&
                    it.classId >= 0 &&
                    it.classId < classNames.size
        }

        for (detection in filteredDetections) {
            // Get color for this class
            val color = classColors[detection.classId % classColors.size]
            paint.color = Color.rgb(color[0], color[1], color[2])

            // Draw bounding box
            canvas.drawRect(
                detection.box.x.toFloat(),
                detection.box.y.toFloat(),
                (detection.box.x + detection.box.width).toFloat(),
                (detection.box.y + detection.box.height).toFloat(),
                paint
            )

            // Create label text
            val label = "${classNames[detection.classId]}: ${(detection.conf * 100).toInt()}%"

            // Measure text for background rectangle
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            // Define label position
            val labelY = max(detection.box.y.toFloat(), textHeight + 5f)

            // Draw background rectangle for text
            val bgPaint = Paint()
            bgPaint.color = Color.rgb(color[0], color[1], color[2])
            bgPaint.style = Paint.Style.FILL

            canvas.drawRect(
                detection.box.x.toFloat(),
                labelY - textHeight - 5f,
                detection.box.x.toFloat() + textWidth + 10f,
                labelY + 5f,
                bgPaint
            )

            // Draw text
            textPaint.color = Color.WHITE
            canvas.drawText(
                label,
                detection.box.x.toFloat() + 5f,
                labelY - 5f,
                textPaint
            )
        }

        return mutableBitmap
    }

    /**
     * Draws bounding boxes and semi-transparent masks on the provided bitmap
     */
    fun drawDetectionsMask(bitmap: Bitmap, detections: List<Detection>, maskAlpha: Float = 0.4f): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val width = bitmap.width
        val height = bitmap.height

        // Create a mask bitmap for overlay
        val maskBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val maskCanvas = Canvas(maskBitmap)

        // Filter detections to ensure quality results
        val filteredDetections = detections.filter {
            it.conf > CONFIDENCE_THRESHOLD &&
                    it.classId >= 0 &&
                    it.classId < classNames.size
        }

        // Draw filled rectangles on mask bitmap
        for (detection in filteredDetections) {
            val color = classColors[detection.classId % classColors.size]
            val paint = Paint()
            paint.color = Color.argb(
                (255 * maskAlpha).toInt(),
                color[0],
                color[1],
                color[2]
            )
            paint.style = Paint.Style.FILL

            maskCanvas.drawRect(
                detection.box.x.toFloat(),
                detection.box.y.toFloat(),
                (detection.box.x + detection.box.width).toFloat(),
                (detection.box.y + detection.box.height).toFloat(),
                paint
            )
        }

        // Overlay mask on original image
        val canvas = Canvas(mutableBitmap)
        val paint = Paint()
        paint.alpha = (255 * maskAlpha).toInt()
        canvas.drawBitmap(maskBitmap, 0f, 0f, paint)

        // Draw bounding boxes and labels (reusing existing method but with full opacity)
        val mainCanvas = Canvas(mutableBitmap)
        val boxPaint = Paint()
        boxPaint.style = Paint.Style.STROKE
        boxPaint.strokeWidth = max(width, height) * 0.004f

        val textPaint = Paint()
        textPaint.textSize = max(width, height) * 0.02f

        for (detection in filteredDetections) {
            val color = classColors[detection.classId % classColors.size]
            boxPaint.color = Color.rgb(color[0], color[1], color[2])

            // Draw bounding box
            mainCanvas.drawRect(
                detection.box.x.toFloat(),
                detection.box.y.toFloat(),
                (detection.box.x + detection.box.width).toFloat(),
                (detection.box.y + detection.box.height).toFloat(),
                boxPaint
            )

            // Create and draw label
            val label = "${classNames[detection.classId]}: ${(detection.conf * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize

            val labelY = max(detection.box.y.toFloat(), textHeight + 5f)

            val bgPaint = Paint()
            bgPaint.color = Color.rgb(color[0], color[1], color[2])
            bgPaint.style = Paint.Style.FILL

            mainCanvas.drawRect(
                detection.box.x.toFloat(),
                labelY - textHeight - 5f,
                detection.box.x.toFloat() + textWidth + 10f,
                labelY + 5f,
                bgPaint
            )

            textPaint.color = Color.WHITE
            mainCanvas.drawText(
                label,
                detection.box.x.toFloat() + 5f,
                labelY - 5f,
                textPaint
            )
        }

        // Clean up
        maskBitmap.recycle()

        return mutableBitmap
    }

    /**
     * Loads the TFLite model file
     */
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelPath)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Loads class names from a file
     */
    private fun loadClassNames(labelsPath: String): List<String> {
        return context.assets.open(labelsPath).bufferedReader().useLines {
            it.map { line -> line.trim() }.filter { it.isNotEmpty() }.toList()
        }
    }

    /**
     * Generate colors for visualization
     */
    private fun generateColors(numClasses: Int): List<IntArray> {
        val colors = mutableListOf<IntArray>()
        val random = Random(42) // Fixed seed for reproducibility

        for (i in 0 until numClasses) {
            val color = intArrayOf(
                random.nextInt(256),  // R
                random.nextInt(256),  // G
                random.nextInt(256)   // B
            )
            colors.add(color)
        }

        return colors
    }

    /**
     * Get class name for a given class ID
     * @param classId The class ID to get the name for
     * @return The class name or "Unknown" if the ID is invalid
     */
    fun getClassName(classId: Int): String {
        return if (classId >= 0 && classId < classNames.size) {
            classNames[classId]
        } else {
            "Unknown"
        }
    }

    /**
     * Get details about the model's input requirements
     * @return String containing shape and data type information
     */
    fun getInputDetails(): String {
        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        val type = when(inputTensor.dataType()) {
            org.tensorflow.lite.DataType.FLOAT32 -> "FLOAT32"
            org.tensorflow.lite.DataType.UINT8 -> "UINT8"
            else -> "OTHER"
        }
        return "Shape: ${shape.joinToString()}, Type: $type"
    }

    /**
     * Cleanup resources when no longer needed
     */
    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    /**
     * Data classes for detections and bounding boxes
     */
    data class BoundingBox(val x: Int, val y: Int, val width: Int, val height: Int)

    data class Detection(val box: BoundingBox, val conf: Float, val classId: Int)

    /**
     * Helper functions
     */

    /**
     * Letterbox an image to fit a specific size while maintaining aspect ratio
     * Fixed to convert Double to Float for proper type handling
     */
    private fun letterBox(
        image: Mat,
        outImage: Mat,
        newShape: Size,
        color: Scalar = Scalar(114.0, 114.0, 114.0),
        auto: Boolean = true,
        scaleFill: Boolean = false,
        scaleUp: Boolean = true,
        stride: Int = 32
    ) {
        val originalShape = Size(image.cols().toDouble(), image.rows().toDouble())

        // Calculate ratio to fit the image within new shape - convert to Float
        var ratio = min(
            newShape.height / originalShape.height,
            newShape.width / originalShape.width
        ).toFloat()

        // Prevent scaling up if not allowed
        if (!scaleUp) {
            ratio = min(ratio, 1.0f)
        }

        // Calculate new unpadded dimensions
        val newUnpadW = round(originalShape.width * ratio).toInt()
        val newUnpadH = round(originalShape.height * ratio).toInt()

        // Calculate padding - convert to Float
        var dw = (newShape.width - newUnpadW).toFloat()
        var dh = (newShape.height - newUnpadH).toFloat()

        if (auto) {
            // Adjust padding to be multiple of stride
            dw = ((dw % stride) / 2).toFloat()
            dh = ((dh % stride) / 2).toFloat()
        } else if (scaleFill) {
            // Scale to fill without maintaining aspect ratio
            dw = 0.0f
            dh = 0.0f
            Imgproc.resize(image, outImage, newShape)
            return
        }

        // Debug for letterbox calculations
        debug("Letterbox: original=${originalShape.width}x${originalShape.height}, " +
              "new=${newUnpadW}x${newUnpadH}, ratio=$ratio, pad=($dw,$dh)")

        // Calculate padded dimensions
        val padLeft = (dw / 2).toInt()
        val padRight = (dw - padLeft).toInt()
        val padTop = (dh / 2).toInt()
        val padBottom = (dh - padTop).toInt()

        // Resize
        Imgproc.resize(
            image,
            outImage,
            Size(newUnpadW.toDouble(), newUnpadH.toDouble()),
            0.0, 0.0,
            Imgproc.INTER_LINEAR
        )

        // Apply padding
        Core.copyMakeBorder(
            outImage,
            outImage,
            padTop,
            padBottom,
            padLeft,
            padRight,
            Core.BORDER_CONSTANT,
            color
        )
    }

    /**
     * Scale coordinates from model input size to original image size
     * Fixed to convert Double to Float for proper type handling
     */
    private fun scaleCoords(
        imageShape: Size,
        coords: RectF,
        imageOriginalShape: Size,
        clip: Boolean = true
    ): RectF {
        // Calculate gain - explicitly convert Double to Float
        val gain = min(
            imageShape.width / imageOriginalShape.width,
            imageShape.height / imageOriginalShape.height
        ).toFloat()
        
        // Calculate padding - explicitly convert Double to Float
        val padX = ((imageShape.width - imageOriginalShape.width * gain) / 2.0).toFloat()
        val padY = ((imageShape.height - imageOriginalShape.height * gain) / 2.0).toFloat()
        
        // Debug inputs for verification
        debug("Scaling: coords=${coords.left},${coords.top},${coords.right},${coords.bottom}")
        debug("Scaling: shape=${imageShape.width}x${imageShape.height}, originalShape=${imageOriginalShape.width}x${imageOriginalShape.height}")
        debug("Scaling: gain=$gain, pad=($padX,$padY)")
        
        // Scale coordinates back to original image size
        val scaledLeft = (coords.left - padX) / gain
        val scaledTop = (coords.top - padY) / gain
        val scaledRight = (coords.right - padX) / gain
        val scaledBottom = (coords.bottom - padY) / gain
        
        debug("Scaling: result=${scaledLeft},${scaledTop},${scaledRight},${scaledBottom}")
        
        // Create result rectangle with validation
        val result = RectF(
            scaledLeft,
            scaledTop,
            scaledRight,
            scaledBottom
        )
        
        // Clip coordinates to image boundaries if requested
        if (clip) {
            result.left = max(0f, min(result.left, imageOriginalShape.width.toFloat()))
            result.top = max(0f, min(result.top, imageOriginalShape.height.toFloat()))
            result.right = max(0f, min(result.right, imageOriginalShape.width.toFloat()))
            result.bottom = max(0f, min(result.bottom, imageOriginalShape.height.toFloat()))
        }
        
        return result
    }

    /**
     * Clamp a value between min and max
     */
    private fun clamp(value: Float, min: Float, max: Float): Float {
        return when {
            value < min -> min
            value > max -> max
            else -> value
        }
    }

    /**
     * Non-Maximum Suppression implementation to filter redundant boxes
     * Updated to exactly match the C++ implementation
     */
    private fun nonMaxSuppression(
        boxes: List<RectF>,
        scores: List<Float>,
        scoreThreshold: Float,
        iouThreshold: Float,
        indices: MutableList<Int>
    ) {
        indices.clear()

        // Early return if no boxes
        if (boxes.isEmpty()) {
            return
        }

        // Create list of indices sorted by score (highest first)
        val sortedIndices = boxes.indices
            .filter { scores[it] >= scoreThreshold }
            .sortedByDescending { scores[it] }

        if (sortedIndices.isEmpty()) {
            return
        }

        // Calculate areas once
        val areas = boxes.map { (it.right - it.left) * (it.bottom - it.top) }

        // Suppression mask
        val suppressed = BooleanArray(boxes.size) { false }

        // Process boxes in order of decreasing score
        for (i in sortedIndices.indices) {
            val currentIdx = sortedIndices[i]

            if (suppressed[currentIdx]) {
                continue
            }

            // Add current box to valid detections
            indices.add(currentIdx)

            // Get current box coordinates
            val currentBox = boxes[currentIdx]
            val x1Max = currentBox.left
            val y1Max = currentBox.top
            val x2Max = currentBox.right
            val y2Max = currentBox.bottom
            val areaCurrent = areas[currentIdx]

            // Compare with remaining boxes
            for (j in i + 1 until sortedIndices.size) {
                val compareIdx = sortedIndices[j]

                if (suppressed[compareIdx]) {
                    continue
                }

                // Calculate intersection
                val compareBox = boxes[compareIdx]
                val x1 = max(x1Max, compareBox.left)
                val y1 = max(y1Max, compareBox.top)
                val x2 = min(x2Max, compareBox.right)
                val y2 = min(y2Max, compareBox.bottom)

                val interWidth = max(0f, x2 - x1)
                val interHeight = max(0f, y2 - y1)

                if (interWidth <= 0 || interHeight <= 0) {
                    continue
                }

                val intersection = interWidth * interHeight
                val unionArea = areaCurrent + areas[compareIdx] - intersection
                val iou = if (unionArea > 0) intersection / unionArea else 0f

                // Suppress if IoU exceeds threshold
                if (iou > iouThreshold) {
                    suppressed[compareIdx] = true
                }
            }
        }
    }

    /**
     * Debug print function
     */
    private fun debug(message: String) {
        Log.d(TAG, message)
        if (BuildConfig.DEBUG) {
            println("YOLO11Detector: $message")
        }
    }

    // Add ScopedTimer implementation (if missing)
    private class ScopedTimer(private val name: String) {
        private val startTime = SystemClock.elapsedRealtime()

        fun stop() {
            val endTime = SystemClock.elapsedRealtime()
//            debug("$name took ${endTime - startTime} ms")
        }
    }
}
