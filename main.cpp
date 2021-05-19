#include "mbed.h"
#include "mbed_rpc.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"
#include "uLCD_4DGL.h"
#include <cmath>

#include "accelerometer_handler.h"
#include "stm32l475e_iot01_accelero.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define NON_MODE 0
#define SET_MODE 1
#define CAP_MODE 2

using namespace std;

uLCD_4DGL uLCD(D1, D0, D2);  // serial tx, serial rx, reset pin;

// GLOBAL VARIABLES
WiFiInterface *wifi;
InterruptIn btn2(USER_BUTTON);
//InterruptIn btn3(SW3);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;

const char *topic = "Mbed";

Thread mqtt_thread(osPriorityHigh);
EventQueue mqtt_queue;

DigitalOut myled1(LED1);
DigitalOut myled2(LED2);
BufferedSerial pc(USBTX, USBRX);
void LEDControl(Arguments *in, Reply *out);
RPCFunction rpcLED(&LEDControl, "set");
int mode;
Thread t1, t2, t3;
double angle_set = 30;
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int16_t DataXYZ[3] = {0};
int16_t G_pDataXYZ[3] = {0};

double thread_hold, tile_r;

void uLCDInit(void) {
    uLCD.background_color(0xFFFFFF);
    uLCD.text_width(3); 
    uLCD.text_height(3);
    uLCD.cls();
}
void uLCDControl(void) {
    uLCD.locate(3,1);
    uLCD.textbackground_color(0xFFFFFF);
    uLCD.color(0x00FF00);
    if (mode == SET_MODE) {
        uLCD.printf("now_select: %lf", angle_set);
    } 
    else if (mode == CAP_MODE) {
        uLCD.printf("now angle: %lf", tile_r);
    } else {
        uLCD.printf("NON MODE", tile_r);
    }
}
void show_mode(void) {
    while(true) {
        printf("now mode: %d\n", mode);
        ThisThread::sleep_for(10000ms);
    }
}
void blinker(int nums)
{
    myled1 = 0;
    ThisThread::sleep_for(1000ms);
    for (int i = 0; i < nums; i++)
    {
        myled1 = 1;
        ThisThread::sleep_for(200ms);
        myled1 = 0;
        ThisThread::sleep_for(200ms);
    }
}
double angle_helper(void)
{

    BSP_ACCELERO_AccGetXYZ(DataXYZ);

    double a = sqrt(pow(G_pDataXYZ[0], 2) + pow(G_pDataXYZ[1], 2) + pow(G_pDataXYZ[2], 2));
    double b = sqrt(pow(DataXYZ[0], 2) + pow(DataXYZ[1], 2) + pow(DataXYZ[2], 2));
    double c = sqrt(pow((DataXYZ[0] - G_pDataXYZ[0]), 2) + pow((DataXYZ[1] - G_pDataXYZ[1]), 2) + pow((DataXYZ[2] - G_pDataXYZ[2]), 2));
    double cos = (a * a + b * b - c * c) / (2 * a * b);
    double theta = acos((pow(a, 2) + pow(b, 2) - pow(c, 2)) / (2 * a * b)) * 180 / 3.1415926;

    return theta;
}
int PredictGesture(float *output)
{
    // How many times the most recent gesture has been matched in a row
    static int continuous_count = 0;
    // The result of the last prediction
    static int last_predict = -1;

    // Find whichever output has a probability > 0.8 (they sum to 1)
    int this_predict = -1;
    thread_hold = 0.8;

    for (int i = 0; i < label_num; i++)
    {
        if (output[i] > thread_hold)
            this_predict = i;
    }

    // No gesture was detected above the threshold
    if (this_predict == -1)
    {
        continuous_count = 0;
        last_predict = label_num;
        return label_num;
    }

    if (last_predict == this_predict)
    {
        continuous_count += 1;
    }
    else
    {
        continuous_count = 0;
    }
    last_predict = this_predict;

    if (continuous_count < config.consecutiveInferenceThresholds[this_predict])
    {
        return label_num;
    }
    continuous_count = 0;
    last_predict = -1;

    return this_predict;
}

int gesture_main(void)
{
    bool should_clear_buffer = false;
    bool got_data = false;

    int gesture_index;

    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter *error_reporter = &micro_error_reporter;

    const tflite::Model *model = tflite::GetModel(g_magic_wand_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        error_reporter->Report(
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static tflite::MicroOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddBuiltin(
        tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                                 tflite::ops::micro::Register_MAX_POOL_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                                 tflite::ops::micro::Register_CONV_2D());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                                 tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                 tflite::ops::micro::Register_SOFTMAX());
    micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                                 tflite::ops::micro::Register_RESHAPE(), 1);

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    tflite::MicroInterpreter *interpreter = &static_interpreter;

    interpreter->AllocateTensors();

    TfLiteTensor *model_input = interpreter->input(0);
    if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
        (model_input->dims->data[1] != config.seq_length) ||
        (model_input->dims->data[2] != kChannelNumber) ||
        (model_input->type != kTfLiteFloat32))
    {
        error_reporter->Report("Bad input tensor parameters in model");
        return -1;
    }

    int input_length = model_input->bytes / sizeof(float);

    TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
    if (setup_status != kTfLiteOk)
    {
        error_reporter->Report("Set up failed\n");
        return -1;
    }

    error_reporter->Report("Set up successful...\n");

    while (true)
    {
        if (mode == SET_MODE) {
            blinker(5);
            printf("gesture_main starts\n");
        }
        while (mode == SET_MODE)
        {
            got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                         input_length, should_clear_buffer);

            if (!got_data)
            {
                should_clear_buffer = false;
                continue;
            }

            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk)
            {
                error_reporter->Report("Invoke failed on index: %d\n", begin_index);
                continue;
            }

            gesture_index = PredictGesture(interpreter->output(0)->data.f);
            should_clear_buffer = gesture_index < label_num;
            if (gesture_index < label_num)
            {
                if (gesture_index == 1 && angle_set > 0)
                { //rtl
                    angle_set -= 10;
                    error_reporter->Report(config.output_message[gesture_index]);
                    printf("angle: %lf\n", angle_set);
                }
                else if (gesture_index == 2 && angle_set < 90)
                { //tr
                    angle_set += 10;
                    error_reporter->Report(config.output_message[gesture_index]);
                    printf("angle: %lf\n", angle_set);
                }
                uLCDControl();
            }
        }
        ThisThread::sleep_for(10000ms);
    }
}

void detect_main(void)
{
    while (true)
    {
        if (mode == CAP_MODE){
            printf("detect_main starts\n");
            blinker(3);
        } 
        while (mode == CAP_MODE)
        {
            tile_r = angle_helper();
            if (tile_r > angle_set)
            {
                printf("reach selected angle %lf\n", angle_set);
                mode = NON_MODE;
            }
            printf("now angle is %lf\n", tile_r);
            uLCDControl();
            ThisThread::sleep_for(300ms);
        }
        ThisThread::sleep_for(10000ms);
        uLCDControl();
    }
}

void messageArrived(MQTT::MessageData &md)
{
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char *)message.payload);
    printf(payload);
    ++arrivedcount;
}

void publish_message_SET(MQTT::Client<MQTTNetwork, Countdown> *client)
{
    if (mode != SET_MODE)
    {
        printf("set to set_mode first");
        return;
    }
    message_num++;
    MQTT::Message message;
    char buff[100];
    sprintf(buff, "QoS0 Hello, Python! #%d, angle selected: %lf\n", message_num, angle_set);
    int tmp = (int)(thread_hold * 10);
    blinker(tmp);
    mode = CAP_MODE;
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void *)buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);
    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
}

void close_mqtt()
{
    closed = true;
}

int main()
{
    mode = NON_MODE;
    uLCDInit();
    uLCDControl();
    BSP_ACCELERO_Init();
    BSP_ACCELERO_AccGetXYZ(G_pDataXYZ);
    wifi = WiFiInterface::get_default_instance();
    if (!wifi)
    {
        printf("ERROR: No WiFiInterface found.\r\n");
        return -1;
    }

    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0)
    {
        printf("\nConnection error: %d\r\n", ret);
        return -1;
    }

    NetworkInterface *net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);

    //TODO: revise host to your IP
    const char *host = "192.168.1.105";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);

    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"), (sockAddr.get_port() ? sockAddr.get_port() : 0)); //check setting

    int rc = mqttNetwork.connect(sockAddr); //(host, 1883);
    if (rc != 0)
    {
        printf("Connection error.");
        return -1;
    }
    printf("Successfully connected!\r\n");

    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";

    if ((rc = client.connect(data)) != 0)
    {
        printf("Fail to connect MQTT\r\n");
    }
    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0)
    {
        printf("Fail to subscribe\r\n");
    }
    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    btn2.rise(mqtt_queue.event(&publish_message_SET, &client));

    char buf[256], outbuf[256];

    FILE *devin = fdopen(&pc, "r");
    FILE *devout = fdopen(&pc, "w");
    t1.start(&gesture_main);
    t2.start(&detect_main);
    t3.start(&show_mode);
    while (1)
    {
        memset(buf, 0, 256); // clear buffer

        for (int i = 0;; i++)
        {
            char recv = fgetc(devin);
            if (recv == '\n')
            {
                printf("\r\n");
                break;
            }
            buf[i] = fputc(recv, devout);
        }
        RPC::call(buf, outbuf);
        printf("%s\r\n", outbuf);
    }

    return 0;
}

void LEDControl(Arguments *in, Reply *out)
{
    mode = SET_MODE;
    printf("Angle setup mode starts\n");
}
