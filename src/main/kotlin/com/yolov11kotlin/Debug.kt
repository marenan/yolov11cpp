package com.yolov11kotlin

/**
 * Debug utility for conditional logging
 */
object Debug {
    /**
     * Print debug message if debug mode is enabled
     */
    fun print(message: Any) {
        if (Config.DEBUG_MODE) {
            println(message)
        }
    }
}
