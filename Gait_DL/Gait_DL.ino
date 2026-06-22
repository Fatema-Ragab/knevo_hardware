#include <Arduino.h>
#include "esp_heap_caps.h"

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n==============================================");
    Serial.println("   ESP32-S3 HARDWARE MEMORY MAP DIAGNOSTIC     ");
    Serial.println("==============================================");

    // Check Total Free Internal SRAM (Where ESP-NN works)
    size_t free_internal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    // FIXED: Correct Espressif function name used here
    size_t max_block_internal = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    Serial.print("Total Free Internal SRAM: ");
    Serial.print(free_internal / 1024.0);
    Serial.println(" KB");
    
    Serial.print("Largest Contiguous Internal SRAM Block: ");
    Serial.print(max_block_internal / 1024.0);
    Serial.println(" KB");
    Serial.println("-> (Your 350KB Tensor Arena needs this block to be > 350 KB)");

    // Check Total Free External PSRAM (For storage/secondary arrays if needed)
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    Serial.print("Total Free External PSRAM (SPIRAM): ");
    Serial.print(free_psram / (1024.0 * 1024.0));
    Serial.println(" MB");
    Serial.println("==============================================");
}

void loop() {
    // Keep empty for diagnostic step
}