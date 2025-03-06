package com.example.opencv_tutorial

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.OpenCVLoader
import java.io.IOException
import java.util.concurrent.Executors
import android.os.SystemClock
import android.graphics.Matrix
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    // Views for UI
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    // YOLOv11 detector instance
    private lateinit var yoloDetector: YOLO11Detector

    // Background thread for async loading
    private val backgroundExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI components
        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)

        // Initialize OpenCV and proceed with detection in background
        initializeOpenCVAndDetector()
    }

    private fun initializeOpenCVAndDetector() {
        resultText.text = "Initializing OpenCV..."

        backgroundExecutor.execute {
            try {
                // Use static initialization for OpenCV
                if (!OpenCVLoader.initDebug()) {
                    Log.e(TAG, "Unable to load OpenCV")
                    runOnUiThread {
                        resultText.text = "Error: OpenCV initialization failed."
                    }
                    return@execute
                }

                // Load native OpenCV library
                try {
                    System.loadLibrary("opencv_java4")
                    Log.i(TAG, "OpenCV loaded successfully")

                    // Now proceed with detector initialization
                    initializeDetectorAndProcess()
                } catch (e: UnsatisfiedLinkError) {
                    Log.e(TAG, "Unable to load OpenCV native library", e)
                    runOnUiThread {
                        resultText.text = "Error: OpenCV native library failed to load.\nError: ${e.message}"
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error during OpenCV initialization", e)
                    runOnUiThread {
                        resultText.text = "Error: ${e.message}"
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error during initialization", e)
                runOnUiThread {
                    resultText.text = "Unexpected error: ${e.message}"
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Reinitialize if necessary but avoid duplicate initialization
        if (!::yoloDetector.isInitialized && !backgroundExecutor.isShutdown) {
            initializeOpenCVAndDetector()
        }
    }

    private fun initializeDetectorAndProcess() {
        runOnUiThread {
            resultText.text = "Loading model and preparing detection..."
        }

        try {
            // Initialize the YOLO11 detector with model and labels from assets
            val modelPath = "best_float16.tflite"
            val labelsPath = "classes.txt"
            val imagePath = "image_2.jpg" // Test image path

            Log.d(TAG, "Loading model from: $modelPath")
            Log.d(TAG, "Loading labels from: $labelsPath")

            // Create detector with GPU acceleration if available
            yoloDetector = YOLO11Detector(
                context = this,
                modelPath = modelPath,
                labelsPath = labelsPath,
                useGPU = true
            )

            runOnUiThread {
                resultText.text = "Model loaded, preparing image..."
            }

            // Load test image from assets with additional debug info
            val imageBitmap = loadImageFromAssets(imagePath)

            if (imageBitmap != null) {
                Log.d(TAG, "Image loaded with dimensions: ${imageBitmap.width}x${imageBitmap.height}")
                
                // Show original image while detection runs
                runOnUiThread {
                    imageView.setImageBitmap(imageBitmap)
                    resultText.text = "Running detection..."
                }

                try {
                    val startTime = SystemClock.elapsedRealtime()

                    // Try with a lower confidence threshold first to see if we detect anything
                    val confThreshold = 0.15f  // Lower than default to catch more detections for debugging
                    val iouThreshold = 0.45f   // Match C++ implementation

                    Log.d(TAG, "Starting detection with conf=$confThreshold, iou=$iouThreshold")

                    // Run detection
                    val detections = yoloDetector.detect(
                        bitmap = imageBitmap,
                        confidenceThreshold = confThreshold,
                        iouThreshold = iouThreshold
                    )

                    val inferenceTime = SystemClock.elapsedRealtime() - startTime
                    Log.d(TAG, "Detection completed in $inferenceTime ms, found ${detections.size} objects")

                    // More detailed logging for debugging
                    if (detections.isEmpty()) {
                        Log.d(TAG, "WARNING: No detections found! Check confidence threshold.")
                    } else {
                        // Log first few detections in more detail
                        detections.take(5).forEachIndexed { index, detection ->
                            val className = yoloDetector.getClassName(detection.classId)
                            val box = detection.box
                            Log.d(TAG, "Top detection #$index: $className (${detection.conf}), " +
                                    "box=${box.x},${box.y},${box.width},${box.height}, " +
                                    "area=${box.width * box.height}")
                        }
                    }

                    // Show results in UI
                    runOnUiThread {
                        if (detections.isNotEmpty()) {
                            // Draw detections with mask overlay for better visualization
                            val resultBitmap = yoloDetector.drawDetectionsMask(imageBitmap, detections)
                            imageView.setImageBitmap(resultBitmap)
                        } else {
                            // Show original image if no detections
                            imageView.setImageBitmap(imageBitmap)
                        }

                        // Format and display detection results
                        val resultInfo = StringBuilder()
                        resultInfo.append("Detection completed in $inferenceTime ms\n")
                        resultInfo.append("Found ${detections.size} objects\n\n")

                        // Display top detections with highest confidence
                        detections.sortedByDescending { it.conf }
                            .take(5)
                            .forEach { detection ->
                                val className = yoloDetector.getClassName(detection.classId)
                                val confidence = (detection.conf * 100).toInt()
                                resultInfo.append("• $className: ${confidence}%\n")
                            }

                        resultText.text = resultInfo.toString()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error during detection", e)
                    // Show original image at least
                    val finalImageBitmap = imageBitmap
                    runOnUiThread {
                        resultText.text = "Detection error: ${e.message}"
                        imageView.setImageBitmap(finalImageBitmap)
                    }
                }
            } else {
                runOnUiThread {
                    resultText.text = "Error: Failed to load image. Please check image_2.jpg in assets folder."
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in detection process", e)
            runOnUiThread {
                resultText.text = "Error: ${e.message}"
            }
        }
    }

    /**
     * Load an image from the assets folder with proper orientation and error handling
     */
    private fun loadImageFromAssets(fileName: String): Bitmap? {
        return try {
            val startTime = SystemClock.elapsedRealtime()

            assets.open(fileName).use { inputStream ->
                val options = BitmapFactory.Options().apply {
                    // Decode at full size for maximum detection quality
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inScaled = false  // Prevent automatic scaling
                }

                val bitmap = BitmapFactory.decodeStream(inputStream, null, options)

                // Log the image loading time and dimensions
                val loadTime = SystemClock.elapsedRealtime() - startTime
                Log.d(TAG, "Image loaded: ${bitmap?.width}x${bitmap?.height}, took $loadTime ms")

                bitmap
            }
        } catch (e: IOException) {
            Log.e(TAG, "Failed to load image '$fileName' from assets", e)
            null
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Out of memory while loading image '$fileName'", e)
            System.gc() // Request garbage collection
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Clean up resources
        if (::yoloDetector.isInitialized) {
            yoloDetector.close()
        }
        // Shutdown executor service
        backgroundExecutor.shutdown()
    }

    companion object {
        private const val TAG = "YOLO11MainActivity"
    }
}
