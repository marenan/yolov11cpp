package com.yolov11kotlin

import android.util.Log

/**
 * Debug utility functions that match the functionality from C++ implementation
 */
object DebugUtils {
    private const val TAG = "YOLO11Debug"
    
    /**
     * Prints a debug message if DEBUG mode is enabled in BuildConfig
     */
    fun debug(message: String) {
        if (BuildConfig.DEBUG) {
            Log.d(TAG, message)
        }
    }
    
    /**
     * Prints an error message regardless of debug mode
     */
    fun error(message: String, throwable: Throwable? = null) {
        if (throwable != null) {
            Log.e(TAG, message, throwable)
        } else {
            Log.e(TAG, message)
        }
    }
    
    /**
     * Prints verbose information about model and inference
     */
    fun logModelInfo(modelPath: String, inputWidth: Int, inputHeight: Int, isQuantized: Boolean, numClasses: Int) {
        if (BuildConfig.DEBUG) {
            Log.d(TAG, "Model: $modelPath")
            Log.d(TAG, "Input dimensions: ${inputWidth}x${inputHeight}")
            Log.d(TAG, "Quantized: $isQuantized")
            Log.d(TAG, "Number of classes: $numClasses")
        }
    }
}
