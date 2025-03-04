import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.5.1"
    application
}

group = "com.yolov11kotlin"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven { url = uri("https://oss.sonatype.org/content/repositories/snapshots") }
}

dependencies {
    // ONNX Runtime
    implementation("com.microsoft.onnxruntime:onnxruntime-mobile:latest.release")
    
    // OpenCV
    implementation("org.openpnp:opencv:4.5.1-2")
    
    // Kotlin standard library
    implementation(kotlin("stdlib"))
    
    // Coroutines for async operations
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.5.0")
    
    // Testing
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnit()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "11"
}

application {
    mainClass.set("com.yolov11kotlin.MainKt")
}

// Task to copy native libraries to the build directory
tasks.register<Copy>("copyNativeLibs") {
    from("libs")
    into("${buildDir}/libs")
    include("**/*.so", "**/*.dll", "**/*.dylib")
}

tasks.named("run") {
    dependsOn("copyNativeLibs")
}
