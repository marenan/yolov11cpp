package com.yolov11kotlin

import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.VideoWriter
import org.opencv.videoio.Videoio
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import kotlin.system.exitProcess

/**
 * Command line parameters for the application
 */
data class CommandLineParams(
    val modelPath: String = "./best.onnx",
    val classesPath: String = "./classes.txt",
    val inputSource: String = "./input.mov",
    val outputPath: String = "./output.mp4",
    val useGPU: Boolean = false,
    val confThreshold: Float = Config.DEFAULT_CONFIDENCE_THRESHOLD,
    val iouThreshold: Float = Config.DEFAULT_IOU_THRESHOLD
)

/**
 * Main application for YOLOv11 video processing
 */
fun main(args: Array<String>) {
    // Load OpenCV native library
    nu.pattern.OpenCV.loadLocally()
    
    // Parse command line arguments
    val params = parseArguments(args)
    
    println("YOLOv11 Kotlin Detector")
    println("---------------------")
    println("Model: ${params.modelPath}")
    println("Classes: ${params.classesPath}")
    println("Input: ${params.inputSource}")
    println("Output: ${params.outputPath}")
    println("GPU: ${params.useGPU}")
    println("Confidence threshold: ${params.confThreshold}")
    println("IoU threshold: ${params.iouThreshold}")
    println("---------------------")
    
    // Initialize YOLO detector
    val detector = try {
        YOLO11Detector(params.modelPath, params.classesPath, params.useGPU)
    } catch (e: Exception) {
        System.err.println("Failed to initialize detector: ${e.message}")
        e.printStackTrace()
        exitProcess(1)
    }
    
    // Open video capture
    val cap = VideoCapture()
    if (params.inputSource.matches(Regex("^\\d+$"))) {
        // Input is a camera index
        cap.open(params.inputSource.toInt())
    } else {
        // Input is a video file
        cap.open(params.inputSource)
    }
    
    if (!cap.isOpened) {
        System.err.println("Error: Could not open the video source!")
        exitProcess(1)
    }
    
    // Get video properties
    val fps = cap.get(Videoio.CAP_PROP_FPS)
    val width = cap.get(Videoio.CAP_PROP_FRAME_WIDTH).toInt()
    val height = cap.get(Videoio.CAP_PROP_FRAME_HEIGHT).toInt()
    
    // Initialize video writer
    val fourcc = VideoWriter.fourcc('m', 'p', '4', 'v') // MP4 codec
    val videoWriter = VideoWriter()
    
    val isWriterOpened = videoWriter.open(
        params.outputPath, 
        fourcc, 
        if (fps > 0) fps else 30.0, 
        Size(width.toDouble(), height.toDouble()), 
        true
    )
    
    if (!isWriterOpened) {
        System.err.println("Error: Could not open video writer!")
        exitProcess(1)
    }
    
    println("Recording output to: ${params.outputPath}")
    println("Press 'q' to stop recording and exit")
    
    // Initialize performance tracking
    var frameCount = 0
    var totalTime = 0.0
    
    val frame = Mat()
    
    while (true) {
        // Read a frame
        if (!cap.read(frame)) {
            println("End of video stream reached")
            break
        }
        
        if (frame.empty()) {
            System.err.println("Error: Could not read a frame!")
            break
        }
        
        // Display input frame
        HighGui.imshow("Input", frame)
        
        // Measure detection time
        val startTime = System.currentTimeMillis()
        
        // Perform detection with the provided thresholds
        val detections = detector.detect(frame, params.confThreshold, params.iouThreshold)
        
        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime
        totalTime += duration
        frameCount++
        
        // Create a copy for output with detections drawn
        val outputFrame = frame.clone()
        
        // Draw bounding boxes and masks on the frame
        detector.drawBoundingBoxMask(outputFrame, detections)
        
        // Add FPS info
        val currentFps = 1000.0 / (totalTime / frameCount)
        Imgproc.putText(
            outputFrame,
            "FPS: ${currentFps.toInt()}",
            Point(20.0, 40.0),
            Imgproc.FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(0.0, 255.0, 0.0),
            2
        )
        
        // Write the processed frame to the output video
        videoWriter.write(outputFrame)
        
        // Display the frame
        HighGui.imshow("Detections", outputFrame)
        
        // Check for 'q' key press to quit (wait 1ms for key)
        val key = HighGui.waitKey(1)
        if (key == 'q'.toInt() || key == 'Q'.toInt()) {
            break
        }
    }
    
    // Release resources
    cap.release()
    videoWriter.release()
    HighGui.destroyAllWindows()
    detector.close()
    
    println("Video processing completed. Output saved to: ${params.outputPath}")
    println("Average FPS: ${1000.0 / (totalTime / frameCount)}")
}

/**
 * Parse command line arguments into parameters
 */
private fun parseArguments(args: Array<String>): CommandLineParams {
    val params = CommandLineParams()
    
    var i = 0
    while (i < args.size) {
        when {
            args[i] == "--model" && i + 1 < args.size -> {
                return params.copy(modelPath = args[i + 1]).also { i += 2 }
            }
            args[i] == "--classes" && i + 1 < args.size -> {
                return params.copy(classesPath = args[i + 1]).also { i += 2 }
            }
            args[i] == "--input" && i + 1 < args.size -> {
                return params.copy(inputSource = args[i + 1]).also { i += 2 }
            }
            args[i] == "--output" && i + 1 < args.size -> {
                return params.copy(outputPath = args[i + 1]).also { i += 2 }
            }
            args[i] == "--conf" && i + 1 < args.size -> {
                try {
                    return params.copy(confThreshold = args[i + 1].toFloat()).also { i += 2 }
                } catch (e: NumberFormatException) {
                    System.err.println("Invalid confidence threshold: ${args[i + 1]}")
                    i += 2
                }
            }
            args[i] == "--iou" && i + 1 < args.size -> {
                try {
                    return params.copy(iouThreshold = args[i + 1].toFloat()).also { i += 2 }
                } catch (e: NumberFormatException) {
                    System.err.println("Invalid IoU threshold: ${args[i + 1]}")
                    i += 2
                }
            }
            args[i] == "--gpu" -> {
                return params.copy(useGPU = true).also { i += 1 }
            }
            else -> {
                System.err.println("Unknown argument: ${args[i]}")
                i += 1
            }
        }
    }
    
    return params
}