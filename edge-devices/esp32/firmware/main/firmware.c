#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "ai_engine.h"

static const char *TAG = "MAIN";

void app_main(void)
{
    ESP_LOGI(TAG, "Starting Edge AI Pipeline Firmware...");
    
    // Initialize the AI Engine
    ai_engine_init();
    
    // Main execution loop
    float sensor_data[6] = {-17.8f, 5.0f, -4.0f, -0.1f, 2.0f, 0.5f};
    float anomaly_score = 0.0f;
    
    while (1) {
        ESP_LOGI(TAG, "Reading sensor data...");
        // Placeholder for reading actual sensors
        
        // Run AI Inference
        ai_engine_predict(sensor_data, &anomaly_score);
        
        ESP_LOGI(TAG, "Anomaly Score: %.2f", anomaly_score);
        
        // Delay for 1 second
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
