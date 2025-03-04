package com.yolov11kotlin

import org.opencv.core.Rect

/**
 * Represents a bounding box with x, y, width, and height coordinates
 */
data class BoundingBox(
    var x: Int,
    var y: Int,
    var width: Int,
    var height: Int
) {
    constructor() : this(0, 0, 0, 0)
    
    fun toRect(): Rect = Rect(x, y, width, height)
    
    companion object {
        fun fromRect(rect: Rect): BoundingBox = BoundingBox(rect.x, rect.y, rect.width, rect.height)
    }
}

/**
 * Represents a detection with bounding box, confidence score, and class ID
 */
data class Detection(
    val box: BoundingBox,
    val conf: Float,
    val classId: Int
)
