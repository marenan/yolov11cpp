package com.yolov11kotlin

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {
    private lateinit var yoloDetector: YOLO11Detector
    
    // Model file names in assets
    private val MODEL_FILE = "best.onnx"
    private val CLASSES_FILE = "classes.txt"
    private val TEST_IMAGE = "image_2.jpg"
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OpenCV
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "Failed to load OpenCV", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // Initialize the YOLO detector
        try {
            // Copy model and class files from assets to app's files directory
            copyAssetToFile(MODEL_FILE)
            copyAssetToFile(CLASSES_FILE)
            copyAssetToFile(TEST_IMAGE)
            
            val modelPath = File(filesDir, MODEL_FILE).absolutePath
            val classesPath = File(filesDir, CLASSES_FILE).absolutePath
            
            // Initialize detector (using CPU for Android compatibility)
            yoloDetector = YOLO11Detector(modelPath, classesPath, useGPU = false)
            Toast.makeText(this, "YOLO model loaded successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load YOLO model: ${e.message}", Toast.LENGTH_LONG).show()
            e.printStackTrace()
            finish()
            return
        }

        // Set up the UI with Compose
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    YoloImageInference(TEST_IMAGE)
                }
            }
        }
    }
    
    @Composable
    fun YoloImageInference(imageName: String) {
        val context = LocalContext.current
        val coroutineScope = rememberCoroutineScope()
        
        // State for the processed image
        var processedBitmap by remember { mutableStateOf<Bitmap?>(null) }
        var isLoading by remember { mutableStateOf(false) }
        var inferenceTime by remember { mutableStateOf(0L) }
        var detectionCount by remember { mutableStateOf(0) }
        
        // Load and process image when the composable is first displayed
        LaunchedEffect(imageName) {
            isLoading = true
            coroutineScope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        // Copy test image from assets if needed
                        copyAssetToFile(imageName)
                        val imagePath = File(context.filesDir, imageName).absolutePath
                        
                        // Load image using OpenCV
                        val imageMat = loadImage(imagePath)
                        if (imageMat == null) {
                            throw IOException("Failed to load image: $imagePath")
                        }
                        
                        // Measure inference time
                        val startTime = System.currentTimeMillis()
                        
                        // Run object detection
                        val detections = yoloDetector.detect(
                            imageMat, 
                            Config.DEFAULT_CONFIDENCE_THRESHOLD, 
                            Config.DEFAULT_IOU_THRESHOLD
                        )
                        
                        inferenceTime = System.currentTimeMillis() - startTime
                        detectionCount = detections.size
                        
                        // Draw bounding boxes
                        val resultMat = imageMat.clone()
                        yoloDetector.drawBoundingBoxMask(resultMat, detections)
                        
                        // Convert Mat to Bitmap
                        val resultBitmap = Bitmap.createBitmap(
                            resultMat.cols(), resultMat.rows(), Bitmap.Config.ARGB_8888
                        )
                        Utils.matToBitmap(resultMat, resultBitmap)
                        
                        // Cleanup
                        imageMat.release()
                        resultMat.release()
                        
                        processedBitmap = resultBitmap
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    withContext(Dispatchers.Main) {
                        Toast.makeText(context, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                    }
                } finally {
                    isLoading = false
                }
            }
        }
        
        // UI for displaying the image and results
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "YOLOv11 Object Detection",
                style = MaterialTheme.typography.h5,
                modifier = Modifier.padding(bottom = 16.dp)
            )
            
            if (isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.padding(24.dp)
                )
                Text("Processing image...")
            } else if (processedBitmap != null) {
                Image(
                    bitmap = processedBitmap!!.asImageBitmap(),
                    contentDescription = "Processed image with detections",
                    modifier = Modifier
                        .weight(1f)
                        .padding(8.dp)
                )
                
                Text("Inference time: $inferenceTime ms")
                Text("Detected objects: $detectionCount")
                
                Button(
                    onClick = { finish() },
                    modifier = Modifier.padding(top = 16.dp)
                ) {
                    Text("Close")
                }
            } else {
                Text("Failed to load or process image")
                
                Button(
                    onClick = { finish() },
                    modifier = Modifier.padding(top = 16.dp)
                ) {
                    Text("Close")
                }
            }
        }
    }
    
    /**
     * Load an image file as an OpenCV Mat
     */
    private fun loadImage(imagePath: String): Mat? {
        try {
            val bitmap = BitmapFactory.decodeFile(imagePath)
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)
            return mat
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }
    
    /**
     * Copy asset file to app's file directory
     */
    private fun copyAssetToFile(assetName: String) {
        try {
            val outFile = File(filesDir, assetName)
            
            // Skip if file already exists
            if (outFile.exists()) {
                return
            }
            
            assets.open(assetName).use { inputStream ->
                FileOutputStream(outFile).use { outputStream ->
                    val buffer = ByteArray(1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
            throw IOException("Failed to copy asset file: $assetName", e)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        // Clean up resources
        if (::yoloDetector.isInitialized) {
            yoloDetector.close()
        }
    }
}