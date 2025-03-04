package com.yolov11kotlin

import ai.onnxruntime.*
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.io.File
import java.lang.Float.MIN_VALUE
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YOLOv11 object detector using ONNX Runtime
 */
class YOLO11Detector(
    private val modelPath: String,
    private val labelsPath: String,
    private val useGPU: Boolean = false
) {
    // OnnxRuntime environment and session
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val isDynamicInputShape: Boolean
    private val inputImageShape: Size
    
    // Class names and colors
    private val classNames: List<String>
    private val classColors: List<Scalar>
    
    init {
        // Initialize ONNX runtime session with appropriate options
        val sessionOptions = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL)
            // Set thread count based on available processors
            setIntraOpNumThreads(min(6, Runtime.getRuntime().availableProcessors()))
            
            // Configure execution providers
            if (useGPU) {
                try {
                    addCUDA()
                    println("Inference device: GPU")
                } catch (e: Exception) {
                    println("GPU is not supported. Fallback to CPU. Error: ${e.message}")
                }
            } else {
                println("Inference device: CPU")
            }
        }
        
        // Load ONNX model
        session = env.createSession(modelPath, sessionOptions)
        
        // Get input details
        val inputInfo = session.inputInfo.values.first()
        val inputShape = (inputInfo.info as TensorInfo).shape
        
        // Determine if the model has dynamic input shape
        isDynamicInputShape = inputShape.size >= 4 && (inputShape[2] == -1L || inputShape[3] == -1L)
        
        // Set input image shape from model info or use default size if dynamic
        inputImageShape = if (isDynamicInputShape) {
            Size(640.0, 640.0)  // Default size for YOLO models
        } else {
            Size(inputShape[3].toDouble(), inputShape[2].toDouble())
        }
        
        // Load class names
        classNames = loadClassNames(labelsPath)
        
        // Generate colors for each class
        classColors = generateColors(classNames)
        
        println("Model loaded successfully with ${session.inputNames.size} inputs and ${session.outputNames.size} outputs.")
    }
    
    /**
     * Detect objects in the provided image
     */
    fun detect(image: Mat, confThreshold: Float = Config.DEFAULT_CONFIDENCE_THRESHOLD, 
               iouThreshold: Float = Config.DEFAULT_IOU_THRESHOLD): List<Detection> {
        return ScopedTimer.measure("Overall detection") {
            // Check for empty images
            if (image.empty()) {
                System.err.println("Error: Empty image provided to detector")
                return@measure emptyList()
            }
            
            // Preprocess the image
            val (preprocessedImage, inputTensor) = preprocess(image)
            
            // Run inference
            val output = runInference(inputTensor)
            
            // Postprocess the output
            val resizedShape = Size(inputImageShape.width, inputImageShape.height)
            postprocess(image.size(), resizedShape, output, confThreshold, iouThreshold)
        }
    }
    
    /**
     * Preprocess the image for inference
     */
    private fun preprocess(image: Mat): Pair<Mat, OnnxTensor> {
        return ScopedTimer.measure("Preprocessing") {
            val resizedImage = Mat()
            
            // Resize and pad the image
            letterbox(image, resizedImage, inputImageShape)
            
            // Convert BGR to RGB (YOLO expects RGB input)
            val rgbImage = Mat()
            Imgproc.cvtColor(resizedImage, rgbImage, Imgproc.COLOR_BGR2RGB)
            
            // Normalize to 0-1
            val normalizedImage = Mat()
            rgbImage.convertTo(normalizedImage, CvType.CV_32FC3, 1.0f/255.0f)
            
            // Convert to tensor format (NCHW)
            val channels = ArrayList<Mat>()
            Core.split(normalizedImage, channels)
            
            // Create float buffer for the tensor
            val inputTensorShape = longArrayOf(1, 3, normalizedImage.rows().toLong(), normalizedImage.cols().toLong())
            val tensorSize = inputTensorShape.reduce { acc, i -> acc * i }.toInt()
            val floatBuffer = FloatBuffer.allocate(tensorSize)
            
            // Copy the data from Mat to FloatBuffer in CHW format
            for (c in 0 until 3) {
                val channel = channels[c]
                for (h in 0 until channel.rows()) {
                    for (w in 0 until channel.cols()) {
                        floatBuffer.put(channel.get(h, w)[0].toFloat())
                    }
                }
            }
            
            floatBuffer.rewind()
            
            // Create ONNXTensor from buffer
            val inputTensor = OnnxTensor.createTensor(env, floatBuffer, inputTensorShape)
            
            Debug.print("Preprocessing completed with RGB conversion")
            
            Pair(resizedImage, inputTensor)
        }
    }
    
    /**
     * Run model inference
     */
    private fun runInference(inputTensor: OnnxTensor): OnnxTensor {
        return ScopedTimer.measure("Inference") {
            val inputName = session.inputNames.iterator().next()
            val inputs = mapOf(inputName to inputTensor)
            
            // Run inference
            val results = session.run(inputs)
            
            // Get output tensor
            val outputTensor = results.get(0) as OnnxTensor
            
            Debug.print("Inference completed")
            outputTensor
        }
    }
    
    /**
     * Postprocess model output to get detection results
     */
    private fun postprocess(
        originalImageSize: Size,
        resizedImageShape: Size,
        outputTensor: OnnxTensor,
        confThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        return ScopedTimer.measure("Postprocessing") {
            val detections = mutableListOf<Detection>()
            
            // Get output shape and data
            val outputShape = outputTensor.info.shape
            val outputData = outputTensor.floatBuffer
            
            // YOLOv11 outputs (batch_size, num_features, num_detections)
            val numFeatures = outputShape[1].toInt()
            val numDetections = outputShape[2].toInt()
            
            if (numDetections == 0) {
                return@measure detections
            }
            
            // Calculate number of classes (output features minus 4 box coordinates)
            val numClasses = numFeatures - 4
            if (numClasses <= 0) {
                return@measure detections
            }
            
            // Prepare lists for NMS
            val boxes = ArrayList<BoundingBox>()
            val scores = ArrayList<Float>()
            val classIds = ArrayList<Int>()
            val nmsBoxes = ArrayList<BoundingBox>()
            
            // Process each detection
            for (d in 0 until numDetections) {
                // Get box coordinates (center_x, center_y, width, height)
                val centerX = outputData.get(0 * numDetections + d)
                val centerY = outputData.get(1 * numDetections + d)
                val width = outputData.get(2 * numDetections + d)
                val height = outputData.get(3 * numDetections + d)
                
                // Find class with highest confidence
                var maxScore = MIN_VALUE
                var classId = -1
                
                for (c in 0 until numClasses) {
                    val score = outputData.get((4 + c) * numDetections + d)
                    if (score > maxScore) {
                        maxScore = score
                        classId = c
                    }
                }
                
                // Filter by confidence threshold
                if (maxScore > confThreshold) {
                    // Convert from center format to top-left format
                    val left = (centerX - width * 0.5f) * resizedImageShape.width.toFloat()
                    val top = (centerY - height * 0.5f) * resizedImageShape.height.toFloat()
                    val w = width * resizedImageShape.width.toFloat()
                    val h = height * resizedImageShape.height.toFloat()
                    
                    // Scale coordinates to original image size
                    val scaledBox = scaleCoords(
                        resizedImageShape,
                        BoundingBox(left.roundToInt(), top.roundToInt(), w.roundToInt(), h.roundToInt()),
                        originalImageSize,
                        true
                    )
                    
                    // Add to vectors for NMS processing
                    val nmsBox = BoundingBox(
                        scaledBox.x + classId * 4096,
                        scaledBox.y + classId * 4096,
                        scaledBox.width,
                        scaledBox.height
                    )
                    
                    boxes.add(scaledBox)
                    nmsBoxes.add(nmsBox)
                    scores.add(maxScore)
                    classIds.add(classId)
                }
            }
            
            // Apply NMS
            val indices = nmsBoxes(nmsBoxes, scores, confThreshold, iouThreshold)
            
            // Create final detections
            for (idx in indices) {
                detections.add(
                    Detection(
                        boxes[idx],
                        scores[idx],
                        classIds[idx]
                    )
                )
            }
            
            Debug.print("Postprocessing completed with ${detections.size} detections after NMS")
            
            detections
        }
    }
    
    /**
     * Draw bounding boxes and masks on the image
     */
    fun drawBoundingBoxMask(image: Mat, detections: List<Detection>, maskAlpha: Float = 0.4f) {
        // Validate input image
        if (image.empty()) {
            System.err.println("ERROR: Empty image provided to drawBoundingBoxMask.")
            return
        }

        val imgHeight = image.rows()
        val imgWidth = image.cols()

        // Precompute dynamic font size and thickness based on image dimensions
        val fontSize = min(imgHeight, imgWidth) * 0.0007
        val textThickness = max(1, (min(imgHeight, imgWidth) * 0.001).toInt())

        // Create a mask image for blending (initialized to zero)
        val maskImage = Mat.zeros(image.size(), image.type())

        // Pre-filter detections
        val filteredDetections = detections.filter { detection ->
            detection.conf > Config.DEFAULT_CONFIDENCE_THRESHOLD && 
            detection.classId >= 0 && 
            detection.classId < classNames.size
        }

        // Draw filled rectangles on the mask image
        for (detection in filteredDetections) {
            val rect = Rect(detection.box.x, detection.box.y, detection.box.width, detection.box.height)
            val color = classColors[detection.classId % classColors.size]
            Imgproc.rectangle(maskImage, rect, color, -1)
        }

        // Blend the maskImage with the original image
        Core.addWeighted(maskImage, maskAlpha.toDouble(), image, 1.0, 0.0, image)

        // Draw bounding boxes and labels
        for (detection in filteredDetections) {
            val rect = Rect(detection.box.x, detection.box.y, detection.box.width, detection.box.height)
            val color = classColors[detection.classId % classColors.size]
            
            // Draw rectangle
            Imgproc.rectangle(image, rect, color, 2)

            // Create label text
            val label = "${classNames[detection.classId]}: ${(detection.conf * 100).toInt()}%"
            val baseline = IntArray(1)
            val labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, fontSize, textThickness, baseline)

            val labelY = max(detection.box.y, labelSize.height + 5)
            val labelTopLeft = Point(detection.box.x.toDouble(), (labelY - labelSize.height - 5).toDouble())
            val labelBottomRight = Point((detection.box.x + labelSize.width + 5).toDouble(), (labelY + baseline[0] - 5).toDouble())

            // Draw background rectangle for label
            Imgproc.rectangle(image, labelTopLeft, labelBottomRight, color, -1)

            // Put label text
            Imgproc.putText(
                image, 
                label, 
                Point(detection.box.x + 2.0, labelY - 2.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                fontSize,
                Scalar(255.0, 255.0, 255.0),
                textThickness
            )
        }

        Debug.print("Bounding boxes and masks drawn on image.")
    }
    
    // ---------- Utility Methods ----------
    
    /**
     * Resize image with letterboxing to maintain aspect ratio
     */
    private fun letterbox(
        image: Mat, 
        outImage: Mat, 
        newShape: Size,
        color: Scalar = Scalar(114.0, 114.0, 114.0),
        auto: Boolean = true,
        scaleFill: Boolean = false,
        scaleUp: Boolean = true,
        stride: Int = 32
    ) {
        // Calculate scaling ratio
        val ratio = min(
            newShape.height / image.rows(),
            newShape.width / image.cols()
        ).toFloat()
        
        // Prevent scaling up if not allowed
        val r = if (!scaleUp) min(ratio, 1.0f) else ratio
        
        // Calculate new unpadded dimensions
        val newUnpadW = (image.cols() * r).roundToInt()
        val newUnpadH = (image.rows() * r).roundToInt()
        
        // Calculate padding
        var dw = (newShape.width - newUnpadW).toInt()
        var dh = (newShape.height - newUnpadH).toInt()
        
        // Adjust padding if needed
        if (auto) {
            dw = (dw % stride) / 2
            dh = (dh % stride) / 2
        } else if (scaleFill) {
            // Scale to fill
            val newUnpadW2 = newShape.width.toInt()
            val newUnpadH2 = newShape.height.toInt()
            Imgproc.resize(image, outImage, Size(newUnpadW2.toDouble(), newUnpadH2.toDouble()))
            return
        }
        
        // Calculate padding for each side
        val padLeft = dw / 2
        val padRight = dw - padLeft
        val padTop = dh / 2
        val padBottom = dh - padTop
        
        // Resize the image
        Imgproc.resize(image, outImage, Size(newUnpadW.toDouble(), newUnpadH.toDouble()))
        
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
     * Scale detection coordinates back to original image size
     */
    private fun scaleCoords(
        imageShape: Size,
        bbox: BoundingBox,
        imageOriginalShape: Size,
        clip: Boolean
    ): BoundingBox {
        // Calculate gain ratio
        val gain = min(
            imageShape.height / imageOriginalShape.height,
            imageShape.width / imageOriginalShape.width
        )
        
        // Calculate padding
        val padX = ((imageShape.width - imageOriginalShape.width * gain) / 2.0).roundToInt()
        val padY = ((imageShape.height - imageOriginalShape.height * gain) / 2.0).roundToInt()
        
        // Adjust coordinates
        val result = BoundingBox()
        result.x = ((bbox.x - padX) / gain).roundToInt()
        result.y = ((bbox.y - padY) / gain).roundToInt()
        result.width = (bbox.width / gain).roundToInt()
        result.height = (bbox.height / gain).roundToInt()
        
        // Clip to image boundaries if requested
        if (clip) {
            result.x = result.x.coerceIn(0, imageOriginalShape.width.toInt())
            result.y = result.y.coerceIn(0, imageOriginalShape.height.toInt())
            result.width = result.width.coerceIn(0, imageOriginalShape.width.toInt() - result.x)
            result.height = result.height.coerceIn(0, imageOriginalShape.height.toInt() - result.y)
        }
        
        return result
    }
    
    /**
     * Non-Maximum Suppression algorithm
     */
    private fun nmsBoxes(
        bboxes: List<BoundingBox>,
        scores: List<Float>,
        scoreThreshold: Float,
        nmsThreshold: Float
    ): List<Int> {
        if (bboxes.isEmpty()) {
            Debug.print("No bounding boxes to process in NMS")
            return emptyList()
        }
        
        // Create indices sorted by score
        val indices = scores.indices
            .filter { scores[it] >= scoreThreshold }
            .sortedByDescending { scores[it] }
            .toMutableList()
            
        if (indices.isEmpty()) {
            Debug.print("No bounding boxes above score threshold")
            return emptyList()
        }
        
        // Calculate areas for all boxes
        val areas = bboxes.map { it.width * it.height.toFloat() }
        
        // Keep track of boxes to suppress
        val suppressed = BooleanArray(bboxes.size)
        
        val keep = mutableListOf<Int>()
        
        var i = 0
        while (i < indices.size) {
            val currentIdx = indices[i]
            
            if (suppressed[currentIdx]) {
                i++
                continue
            }
            
            keep.add(currentIdx)
            
            // Box coordinates
            val currentBox = bboxes[currentIdx]
            val x1Max = currentBox.x
            val y1Max = currentBox.y
            val x2Max = currentBox.x + currentBox.width
            val y2Max = currentBox.y + currentBox.height
            val areaCurrent = areas[currentIdx]
            
            // Compare with remaining boxes
            for (j in i + 1 until indices.size) {
                val idx = indices[j]
                if (suppressed[idx]) continue
                
                val box = bboxes[idx]
                val x1 = max(x1Max, box.x)
                val y1 = max(y1Max, box.y)
                val x2 = min(x2Max, box.x + box.width)
                val y2 = min(y2Max, box.y + box.height)
                
                val interWidth = max(0, x2 - x1)
                val interHeight = max(0, y2 - y1)
                
                if (interWidth <= 0 || interHeight <= 0) continue
                
                val intersection = interWidth * interHeight.toFloat()
                val union = areaCurrent + areas[idx] - intersection
                val iou = if (union > 0) intersection / union else 0f
                
                if (iou > nmsThreshold) {
                    suppressed[idx] = true
                }
            }
            i++
        }
        
        Debug.print("NMS completed with ${keep.size} indices remaining")
        return keep
    }
    
    /**
     * Load class names from file
     */
    private fun loadClassNames(filePath: String): List<String> {
        val classNames = mutableListOf<String>()
        try {
            File(filePath).useLines { lines ->
                for (line in lines) {
                    val trimmed = line.trim()
                    if (trimmed.isNotEmpty()) {
                        classNames.add(trimmed)
                    }
                }
            }
        } catch (e: Exception) {
            System.err.println("ERROR: Failed to access class name path: $filePath")
            e.printStackTrace()
        }
        
        Debug.print("Loaded ${classNames.size} class names from $filePath")
        return classNames
    }
    
    /**
     * Generate random colors for visualization
     */
    private fun generateColors(classNames: List<String>): List<Scalar> {
        val random = java.util.Random(42)
        return classNames.map {
            Scalar(
                random.nextInt(256).toDouble(),
                random.nextInt(256).toDouble(),
                random.nextInt(256).toDouble()
            )
        }
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        session.close()
        env.close()
    }
}
