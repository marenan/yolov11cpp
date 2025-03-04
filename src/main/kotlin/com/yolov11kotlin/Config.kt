package com.yolov11kotlin

/**
 * Configuration settings for the YOLOv11 detector
 */
object Config {
    // Enable debug mode for additional console output
    const val DEBUG_MODE = true
    
    // Enable timing measurements
    const val TIMING_MODE = true
    
    // Default confidence threshold
    const val DEFAULT_CONFIDENCE_THRESHOLD = 0.25f
    
    // Default IoU threshold for NMS
    const val DEFAULT_IOU_THRESHOLD = 0.45f
}
