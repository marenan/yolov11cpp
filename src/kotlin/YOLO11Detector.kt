package com.yolov11kotlin

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
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
    // Detection parameters
    companion object {
        const val CONFIDENCE_THRESHOLD = 0.4f
        const val IOU_THRESHOLD = 0.3f
    }

    // Data structures for model and inference
    private var interpreter: Interpreter
    private val classNames: List<String>
    private val classColors: List<IntArray>

    // Input shape info
    private var inputWidth: Int = 640
    private var inputHeight: Int = 640
    private var isQuantized: Boolean = false

    init {
        // Load model
        val tfliteOptions = Interpreter.Options()

        // GPU Delegate setup if available and requested
        if (useGPU) {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                debug("GPU acceleration enabled")
                val delegateOptions = compatList.bestOptionsForThisDevice
                val gpuDelegate = GpuDelegate(delegateOptions)
                tfliteOptions.addDelegate(gpuDelegate)
            } else {
                debug("GPU acceleration not supported, using CPU")
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

        debug("Model loaded with input dimensions: $inputWidth x $inputHeight")
        debug("Model uses ${if(isQuantized) "quantized" else "float"} input")

        // Load class names and generate colors
        classNames = loadClassNames(labelsPath)
        classColors = generateColors(classNames.size)

        debug("Loaded ${classNames.size} classes")
    }

    /**
     * Main detection function that processes an image and returns detected objects
     */
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = CONFIDENCE_THRESHOLD,
               iouThreshold: Float = IOU_THRESHOLD): List<Detection> {
        val startTime = SystemClock.elapsedRealtime()

        // Convert Bitmap to Mat for OpenCV processing
        val inputMat = Mat()
        Utils.bitmapToMat(bitmap, inputMat)
        Imgproc.cvtColor(inputMat, inputMat, Imgproc.COLOR_RGBA2BGR)

        // Prepare input for TFLite
        val originalSize = Size(bitmap.width.toDouble(), bitmap.height.toDouble())
        val inputData = prepareInput(inputMat)
        val outputs = runInference(inputData)

        // Process outputs to get detections
        val detections = postprocess(
            outputs,
            originalSize,
            Size(inputWidth.toDouble(), inputHeight.toDouble()),
            confidenceThreshold,
            iouThreshold
        )

        inputMat.release()

        val inferenceTime = SystemClock.elapsedRealtime() - startTime
        debug("Detection completed in $inferenceTime ms with ${detections.size} objects")

        return detections
    }

    /**
     * Prepares the input image for TFLite inference
     */
    private fun prepareInput(image: Mat): ByteBuffer {
        val scopedTimer = ScopedTimer("preprocessing")

        // Create resized image with letterboxing
        val resizedImage = Mat()
        letterBox(image, resizedImage, Size(inputWidth.toDouble(), inputHeight.toDouble()), Scalar(114.0, 114.0, 114.0))

        // Prepare the ByteBuffer to store the model input data
        val bytesPerChannel = if (isQuantized) 1 else 4
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputWidth * inputHeight * 3 * bytesPerChannel)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert to RGB and normalize
        val rgbMat = Mat()
        Imgproc.cvtColor(resizedImage, rgbMat, Imgproc.COLOR_BGR2RGB)

        // Extract pixels and normalize
        val pixels = ByteArray(rgbMat.width() * rgbMat.height() * 3)
        rgbMat.get(0, 0, pixels)

        // Process and convert the pixels
        var pixel = 0
        for (y in 0 until inputHeight) {
            for (x in 0 until inputWidth) {
                for (c in 0 until 3) {
                    val value = pixels[pixel++].toInt() and 0xFF
                    if (isQuantized) {
                        inputBuffer.put(value.toByte())
                    } else {
                        inputBuffer.putFloat(value / 255.0f)
                    }
                }
            }
        }

        inputBuffer.rewind()

        // Clean up OpenCV resources
        resizedImage.release()
        rgbMat.release()

        scopedTimer.stop()
        return inputBuffer
    }

    /**
     * Runs inference with TFLite and returns the raw output
     */
    private fun runInference(inputBuffer: ByteBuffer): Map<Int, Any> {
        val scopedTimer = ScopedTimer("inference")

        // Define outputs based on the model
        val outputs: MutableMap<Int, Any> = HashMap()

        // YOLOv11 outputs a single tensor with shape [1, 4+num_classes, num_detections]
        val outputShape = interpreter.getOutputTensor(0).shape()
        val numFeatures = outputShape[1]
        val numDetections = outputShape[2]

        val outputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(4 * numFeatures * numDetections)
        } else {
            ByteBuffer.allocateDirect(4 * numFeatures * numDetections)
        }
        outputBuffer.order(ByteOrder.nativeOrder())

        outputs[0] = outputBuffer

        // Run inference
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        scopedTimer.stop()
        return outputs
    }

    /**
     * Post-processes the model outputs to extract detections
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

        // Get output buffer
        val outputBuffer = outputMap[0] as ByteBuffer
        outputBuffer.rewind()

        // Get output dimensions
        val outputShapes = interpreter.getOutputTensor(0).shape()
        val numFeatures = outputShapes[1]
        val numDetections = outputShapes[2]

        // Calculate number of classes
        val numClasses = numFeatures - 4

        // Extract boxes, confidences, and class ids
        val boxes = mutableListOf<RectF>()
        val confidences = mutableListOf<Float>()
        val classIds = mutableListOf<Int>()
        val nmsBoxes = mutableListOf<RectF>()

        // Process each detection
        for (d in 0 until numDetections) {
            // Get bounding box coordinates
            val centerX = outputBuffer.getFloat()
            val centerY = outputBuffer.getFloat()
            val width = outputBuffer.getFloat()
            val height = outputBuffer.getFloat()

            // Find class with maximum score
            var maxScore = -Float.MAX_VALUE
            var classId = -1

            for (c in 0 until numClasses) {
                val score = outputBuffer.getFloat()
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

                // Scale coordinates to original image size
                val scaledBox = scaleCoords(
                    resizedImageShape,
                    RectF(left, top, left + width, top + height),
                    originalImageSize
                )

                // Ensure box coordinates are valid
                val roundedBox = RectF(
                    round(scaledBox.left),
                    round(scaledBox.top),
                    round(scaledBox.right),
                    round(scaledBox.bottom)
                )

                // Create offset box for NMS
                val nmsBox = RectF(
                    roundedBox.left + classId * 7680,
                    roundedBox.top + classId * 7680,
                    roundedBox.right + classId * 7680,
                    roundedBox.bottom + classId * 7680
                )

                nmsBoxes.add(nmsBox)
                boxes.add(roundedBox)
                confidences.add(maxScore)
                classIds.add(classId)
            }
        }

        // Run NMS to eliminate redundant boxes
        val selectedIndices = mutableListOf<Int>()
        nonMaxSuppression(nmsBoxes, confidences, confThreshold, iouThreshold, selectedIndices)

        // Create final detection objects
        selectedIndices.forEach { idx ->
            val box = boxes[idx]
            detections.add(
                Detection(
                    BoundingBox(
                        box.left.toInt(),
                        box.top.toInt(),
                        (box.right - box.left).toInt(),
                        (box.bottom - box.top).toInt()
                    ),
                    confidences[idx],
                    classIds[idx]
                )
            )
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

        for (detection in detections) {
            if (detection.conf <= CONFIDENCE_THRESHOLD) continue

            if (detection.classId < 0 || detection.classId >= classNames.size) continue

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

        // Filter detections
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
     * Cleanup resources when no longer needed
     */
    fun close() {
        interpreter.close()
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

        // Calculate ratio to fit the image within new shape
        var ratio = min(
            newShape.height / originalShape.height,
            newShape.width / originalShape.width
        )

        // Prevent scaling up if not allowed
        if (!scaleUp) {
            ratio = min(ratio, 1.0)
        }

        // Calculate new unpadded dimensions
        val newUnpadW = round(originalShape.width * ratio).toInt()
        val newUnpadH = round(originalShape.height * ratio).toInt()

        // Calculate padding
        var dw = newShape.width - newUnpadW
        var dh = newShape.height - newUnpadH

        if (auto) {
            // Adjust padding to be multiple of stride
            dw = (dw % stride) / 2
            dh = (dh % stride) / 2
        } else if (scaleFill) {
            // Scale to fill without maintaining aspect ratio
            dw = 0.0
            dh = 0.0
            Imgproc.resize(image, outImage, newShape)
            return
        }

        // Calculate padded dimensions
        val padLeft = (dw / 2).toInt()
        val padRight = (dw - padLeft).toInt()
        val padTop = (dh / 2).toInt()
        val padBottom = (dh - padTop).toInt()

        // Resize
        Imgproc.resize(
            image,
            outImage,
            Size(newUnpadW.toDouble(), newUnpadH.toDouble())
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
     */
    private fun scaleCoords(
        imageShape: Size,
        coords: RectF,
        imageOriginalShape: Size,
        clip: Boolean = true
    ): RectF {
        // Calculate gain based on aspect ratio
        val gain = min(
            imageShape.height / imageOriginalShape.height,
            imageShape.width / imageOriginalShape.width
        )

        // Calculate padding
        val padX = (imageShape.width - imageOriginalShape.width * gain) / 2
        val padY = (imageShape.height - imageOriginalShape.height * gain) / 2

        // Scale coordinates back to original image size
        val scaledLeft = (coords.left - padX) / gain
        val scaledTop = (coords.top - padY) / gain
        val scaledRight = (coords.right - padX) / gain
        val scaledBottom = (coords.bottom - padY) / gain

        // Clip coordinates to image boundaries if requested
        val result = RectF(scaledLeft.toFloat(), scaledTop.toFloat(),
            scaledRight.toFloat().toFloat(), scaledBottom.toFloat())

        if (clip) {
            result.left = clamp(result.left, 0f, imageOriginalShape.width.toFloat())
            result.top = clamp(result.top, 0f, imageOriginalShape.height.toFloat())
            result.right = clamp(result.right, 0f, imageOriginalShape.width.toFloat())
            result.bottom = clamp(result.bottom, 0f, imageOriginalShape.height.toFloat())
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

        // Create list of indices sorted by score
        val sortedIndices = boxes.indices
            .filter { scores[it] >= scoreThreshold }
            .sortedByDescending { scores[it] }

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
        if (BuildConfig.DEBUG) {
            println("YOLO11Detector: $message")
        }
    }
}
