#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model.h"
#include <Wire.h>
#include <MPU6050_light.h>

#define NUM_MPUS 3
#define SAMPLE_RATE_HZ 64
#define BUFFER_SIZE 128
#define FEATURE_COUNT 9
#define OVERLAP_RATIO 0.5
#define LED_PIN 2

const int AD0pin[NUM_MPUS] = {16, 17, 18};
const int mpuAddr = 0x68;
MPU6050 mpu[NUM_MPUS] = {MPU6050(Wire), MPU6050(Wire), MPU6050(Wire)};

float sensor_buffer[BUFFER_SIZE][FEATURE_COUNT];
int buffer_index = 0;
//int iterator=0;
const int step_size = BUFFER_SIZE * (1 - OVERLAP_RATIO);
unsigned long previousMillis = 0;
const unsigned long interval = 1000 / SAMPLE_RATE_HZ;
int start_ind=0;

// TensorFlow Lite variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
uint8_t tensor_arena[32 * 1024];
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setupTFLite() {
    tflite_model = tflite::GetModel(model_data);
    if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema!");
        while (true);
    }
    static tflite::MicroInterpreter static_interpreter(tflite_model, resolver, tensor_arena, sizeof(tensor_arena), &micro_error_reporter);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Tensor allocation failed!");
        while (true);
    }
    input_tensor = interpreter->input(0);
    output_tensor = interpreter->output(0);
    Serial.println("TensorFlow Lite initialized successfully.");
}

void SelectMPU(int selection) {
    for (int i = 0; i < NUM_MPUS; i++) {
        digitalWrite(AD0pin[i], i == selection ? LOW : HIGH);
    }
}

void readMPUData() {
    for (int i = 0; i < NUM_MPUS; i++) {
        SelectMPU(i);
        mpu[i].update();
        //delay(100);
        float val=random(-4000,4000);
        sensor_buffer[buffer_index][i * 3] = mpu[i].getAccX()*1000+val;
        sensor_buffer[buffer_index][i * 3 + 1] = mpu[i].getAccY()*1000+val;
        sensor_buffer[buffer_index][i * 3 + 2] = mpu[i].getAccZ()*1000+val;
    }
    buffer_index = (buffer_index + 1) % BUFFER_SIZE;
    //iterator++;
}

/*void runModel() {
    float input_data[BUFFER_SIZE * FEATURE_COUNT];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        for (int j = 0; j < FEATURE_COUNT; j++) {
            input_data[i * FEATURE_COUNT + j] = sensor_buffer[(buffer_index + i) % BUFFER_SIZE][j];
        }
    }
    memcpy(input_tensor->data.f, input_data, sizeof(input_data));
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Model invocation failed!");
        return;
    }
    float prob = output_tensor->data.f[0];
    float threshold=0.36493510007858276;
    // float class_1_prob = output_tensor->data.f[1];
    int predicted_class = (prob < threshold ) ? 0 : 1;
    digitalWrite(LED_PIN, predicted_class);
    Serial.printf("Probability that fog has occured: %.2f",prob);
    Serial.println();
    if(predicted_class==1){
        Serial.printf("FoG EPISODE DETECTED");
    } else{
        Serial.printf("NO FoG EPISODE DETECTED");
    }
}*/
void runModel() {
    float input_data[BUFFER_SIZE * FEATURE_COUNT];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        for (int j = 0; j < FEATURE_COUNT; j++) {
            input_data[i * FEATURE_COUNT + j] = sensor_buffer[(start_ind + i) % BUFFER_SIZE][j] ;
        }
    }
    
    // Print the first few values to check if the data is valid
    /*Serial.println("Input Data:");
    for (int i = 0; i < FEATURE_COUNT; i++) {
        Serial.printf("%.2f ", input_data[i]);
    }
    Serial.println();*/

    memcpy(input_tensor->data.f, input_data, sizeof(input_data));

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Model invocation failed!");
        return;
    }

    float prob = output_tensor->data.f[0];
    float threshold = 0.7539649605751038;
    int predicted_class = (prob < threshold) ? 0 : 1;

    digitalWrite(LED_PIN, predicted_class);
    Serial.printf("Probablity: %.4f  ", prob);
    Serial.println();

    if (predicted_class == 1) {
        Serial.println("FoG EPISODE DETECTED\n");
    } else {
        Serial.println("NO FoG EPISODE DETECTED\n");
    }
}


void setup() {
    Serial.begin(115200);
    //Wire.begin();
    Wire.begin(21,22);
    Wire.setClock(100000);
    /*for (int i = 0; i < BUFFER_SIZE; i++) {
        for (int j = 0; j < FEATURE_COUNT; j++) {
            sensor_buffer[i][j] = random(-6000, 6000);  
        }
    }*/
    for (int i = 0; i < NUM_MPUS; i++) {
        pinMode(AD0pin[i], OUTPUT);
        digitalWrite(AD0pin[i], HIGH);
        SelectMPU(i);
        mpu[i].begin();
        delay(100);
    }
    pinMode(LED_PIN, OUTPUT);
    setupTFLite();
    previousMillis = millis();
    Serial.println("Setup complete.");
}

void loop() {
    if (millis() - previousMillis >= interval) {
        previousMillis = millis();
        readMPUData();
        if (buffer_index % step_size == 0) {
            if(buffer_index==step_size){
                start_ind=0;
            }
            else{
                start_ind=step_size;
            }
            runModel();
            Serial.println();
        }
    }
    //delay(50);
}