/* Script reads data from Prototype device  to serial in JSON Format over BlueTooth
  Sensors include Omron, Sharp IR sensor, IMU, and FSRs.

  Multiplexer setup is used to access the OMRON
  This script reads the data from a 8x1 OMRON D6T-8L sensor using only the Wire library.
  Note that use of D6T-44L 4x4 sensor requires the WireExt.h library, so that the 35 bytes of data can be read from the sensor.
  Original code from Arduino forum on WireExt.h https://forum.arduino.cc/index.php?topic=217394.0

  Pin Assignments

  c1 Omron 8x1 on mux
  c2 omron 4x4 on mux
  A0, A1 for Sharp IR Sensors

  Update Summary:
  Sept 2017:   Updating to output accelerometer data, and ignore the distal Omron sensors
  Oct 2017: Byte transmission instead of serial.print char
  Dec 17 2017: Increased Accelerometer and Gyro sensitivity to 8G and 1200DPS. Check Sensor API for units and background.
  https://github.com/adafruit/Adafruit_LSM9DS0_Library/blob/master/examples/sensorapi/sensorapi.ino


*/

#include <Wire.h>
#include <WireExt.h>
#include <elapsedMillis.h>
#include <ArduinoJson.h>
#include <TimeLib.h>
#include <Time.h>
#include <SPI.h>

// Variable Setup
int packet = 0;
long runtime = 0;
int DL1;
int DS1;
int DL2;
int DS2;
float AcX, AcY, AcZ;
float GyX, GyY, GyZ;
char O8tol[8];
char O16tol[16];

//First Loop Check
//int LoopOne = 0;


// Logic Checking
float checkSum = 0;
int startStopBytes = 2;
int floatCheckSumBytes = 4;
int numBytes = 7 * 4 + 25 * 2 + startStopBytes + floatCheckSumBytes;
int index = 0; //tracking current position in send buffer

//Bluetooth Pins Setup
#include <SoftwareSerial.h>

int bluetoothTx = 3;  // TX-O pin of bluetooth mate, Arduino D2
int bluetoothRx = 2;  // RX-I pin of bluetooth mate, Arduino D3

SoftwareSerial bluetooth(bluetoothTx, bluetoothRx);

// IMU Setup
#include <Adafruit_LSM9DS0.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_Sensor_Set.h>
#include <Adafruit_Simple_AHRS.h>

Adafruit_LSM9DS0 lsm = Adafruit_LSM9DS0();
Adafruit_Simple_AHRS ahrs(&lsm.getAccel(), &lsm.getMag());

void setupSensor() // Actually have to run the setupSensor() script to initiate at desired ranges
{
  // 1.) Initialize each component
//  lsm.setupAccel(lsm.LSM9DS0_ACCELRANGE_2G);
  lsm.setupAccel(lsm.LSM9DS0_ACCELRANGE_8G);
  lsm.setupMag(lsm.LSM9DS0_MAGGAIN_2GAUSS);
//  lsm.setupGyro(lsm.LSM9DS0_GYROSCALE_245DPS);
  lsm.setupGyro(lsm.LSM9DS0_GYROSCALE_2000DPS);

}

void displaySensorDetails(void)
{
  sensor_t accel, mag, gyro, temp;
  
  lsm.getSensor(&accel, &mag, &gyro, &temp);
  
  Serial.println(F("------------------------------------"));
  Serial.print  (F("Sensor:       ")); Serial.println(accel.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(accel.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(accel.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(accel.max_value); Serial.println(F(" m/s^2"));
  Serial.print  (F("Min Value:    ")); Serial.print(accel.min_value); Serial.println(F(" m/s^2"));
  Serial.print  (F("Resolution:   ")); Serial.print(accel.resolution); Serial.println(F(" m/s^2"));  
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));

  Serial.println(F("------------------------------------"));
  Serial.print  (F("Sensor:       ")); Serial.println(mag.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(mag.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(mag.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(mag.max_value); Serial.println(F(" uT"));
  Serial.print  (F("Min Value:    ")); Serial.print(mag.min_value); Serial.println(F(" uT"));
  Serial.print  (F("Resolution:   ")); Serial.print(mag.resolution); Serial.println(F(" uT"));  
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));

  Serial.println(F("------------------------------------"));
  Serial.print  (F("Sensor:       ")); Serial.println(gyro.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(gyro.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(gyro.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(gyro.max_value); Serial.println(F(" rad/s"));
  Serial.print  (F("Min Value:    ")); Serial.print(gyro.min_value); Serial.println(F(" rad/s"));
  Serial.print  (F("Resolution:   ")); Serial.print(gyro.resolution); Serial.println(F(" rad/s"));  
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));

  Serial.println(F("------------------------------------"));
  Serial.print  (F("Sensor:       ")); Serial.println(temp.name);
  Serial.print  (F("Driver Ver:   ")); Serial.println(temp.version);
  Serial.print  (F("Unique ID:    ")); Serial.println(temp.sensor_id);
  Serial.print  (F("Max Value:    ")); Serial.print(temp.max_value); Serial.println(F(" C"));
  Serial.print  (F("Min Value:    ")); Serial.print(temp.min_value); Serial.println(F(" C"));
  Serial.print  (F("Resolution:   ")); Serial.print(temp.resolution); Serial.println(F(" C"));  
  Serial.println(F("------------------------------------"));
  Serial.println(F(""));
  
  delay(500);
}


// End of IMU Setup


//elapsedMillis timeElapsed; // Global variable declaration for the elapsed time in the script as variable timeElapsed. Time output is not working yet :(

// Omron Sensor setup~~~~
#define D6T_addr 0x0A // 7 bit address of OMRON D6T is 0x0A in hex, 0000 1010 in binary
#define D6T_cmd 0x4C // Standard command is 4C in hex, 0100 1100 in binary

// Declare memor objects
int numbytes = 19; // omron8
int numel = 8; // omron8
//First Set of Omron Sensors
int rbuf[19]; // Actual raw data is coming in as 35 bytes for OMRON 4x4, and 19 bytes for Omron 8x1.
int tdata[8]; // The data comming from omron8 is in 8 elements and 16 elements for omron4x4
int rbuf2[35]; // Actual raw data is coming in as 35 bytes for OMRON 4x4
int tdata2[16]; // The data comming from the omron4x4 is in 16 elements
float t_PTAT;
int i = 0;
//~~~~~~~~~~~~~~~~~~~~~~~~~

// Multiplexer setup
const int selectPins[3] = {9, 8, 7}; // S0~9, S1~8, S2~7
const int zOutput = 5; // Connect common (Z) to 5 (PWM-capable)

const int LED_ON_TIME = 500; // Each LED is on 0.5s
const int DELAY_TIME = ((float)LED_ON_TIME / 512.0) * 1000;
//~~~~~~~~~~~~~~~~~~~~~~~~~

// Pin setups
// Omron mux pins
const int omron8 = 4; // mux address, 4 on PCB
const int omron4 = 6; // mux address, 6 on PCB
const int omron8_2 = 5; // mux address, 5 on PCB
const int omron4_2 = 7; // mux addres, 7 on PCB

// Sharp Sensor setup
const int ProxL = 3; // The long range IR sensor, 10-80 cm. Analog addres
const int ProxS = 1; // The short range IR sensor, 2-15 cm. Analog address
const int ProxL2 = 2; // The long range IR sensor, 10-80 cm. Analog addres
const int ProxS2 = 0; // The short range IR sensor, 2-15 cm. Analog address


void setup()
{
  // Adafruit IMU Setup
  // We read IMU right from the i2c main line - no mux required.
#ifndef ESP8266
  while (!Serial);     // will pause Zero, Leonardo, etc until serial console opens
#endif


  delay(10);
  Serial.begin(115200);
  //Bluetooth Setup
  bluetooth.begin(115200);  // The Bluetooth Mate defaults to 115200bps
  bluetooth.print("$");  // Print three times individually
  bluetooth.print("$");
  bluetooth.print("$");  // Enter command mode
  delay(100);  // Short delay, wait for the Mate to send back CMD
  bluetooth.println("U,115200,N");  // Temporarily Change the baudrate to 38400, no parity
  // 115200 can be too fast at times for NewSoftSerial to relay the data reliably
  bluetooth.begin(115200);  // Start bluetooth serial at 9600

  delay(50);

  //  Serial.println("LSM raw read demo");
  // Try to initialise and warn if we couldn't detect the chip
  Serial.println("Hi!");
  if (!lsm.begin())
  {
    Serial.println("Oops ... unable to initialize the LSM9DS0. Check your wiring!");
    while (1);
    Serial.println("R.I.P.");
  }
  Serial.println("LSM9DS0 9DOF IMU Detected.");
  
  // IMU setup and config
  displaySensorDetails();
  setupSensor(); // Configure the Adafruit LSM IMU Config settings, including acc G range, gyro dps
  
  Serial.println("");

  for (int i = 0; i < 3; i++)
  {
    pinMode(selectPins[i], OUTPUT);
    digitalWrite(selectPins[i], LOW);
  }
  pinMode(zOutput, OUTPUT); // Set up Z as an output

  Wire.begin();
  Serial.println("Hello!");

  selectMuxPin(omron4_2); // Select a single Omron to read from
}

void loop()
{
  acquireData(); // Acquire Sensor Data

  // Create our data object and start writing
  byte bytes[numBytes];
  for (int i = 0; i < numBytes; i++)
  {
    bytes[i] = 0;
  }
  index = 0;
  runtime = millis();
  bytes[index++] = 0x3C; // <
  bytes[index++] = 0x3C;
  bytes[index++] = (packet >> 8) & 0xFF;
  bytes[index++] = packet & 0xFF;
  bytes[index++] = (runtime >> 24) & 0xFF;
  bytes[index++] = (runtime >> 16) & 0xFF;
  bytes[index++] = (runtime >> 8) & 0xFF;
  bytes[index++] = runtime & 0xFF;

  bytes[index++] = (DL1 >> 8) & 0xFF;
  bytes[index++] = DL1 & 0xFF; 
  
  bytes[index++] = (DS1 >> 8) & 0xFF;
  bytes[index++] = DS1 & 0xFF; 
  
  bytes[index++] = (DL2 >> 8) & 0xFF;
  bytes[index++] = DL2 & 0xFF; 
  
  bytes[index++] = (DS2 >> 8) & 0xFF;
  bytes[index++] = DS2 & 0xFF; 
  
  float2Bytes(AcX, &bytes[index]);
  index += 4;
  float2Bytes(AcY, &bytes[index]);
  index += 4;
  float2Bytes(AcZ, &bytes[index]);
  index += 4;
  float2Bytes(GyX,&bytes[index]);
  index += 4;
  float2Bytes(GyY,&bytes[index]);
  index += 4;
  float2Bytes(GyZ,&bytes[index]);
  index += 4;

//  for (int i = 0; i < 8; i++) // Omron8
//  {
//    bytes[index++] = (tdata[i] >> 8) & 0xFF;
//    bytes[index++] = tdata[i] & 0xFF;
//  }
  bytes[index++] = (tdata2[0]) & 0xFF; // Omro4x4
  bytes[index++] = (tdata2[1]) & 0xFF; 
  bytes[index++] = (tdata2[2]) & 0xFF; 
  bytes[index++] = (tdata2[3]) & 0xFF; 
  bytes[index++] = (tdata2[4]) & 0xFF; 
  bytes[index++] = (tdata2[5]) & 0xFF; 
  bytes[index++] = (tdata2[6]) & 0xFF; 
  bytes[index++] = (tdata2[7]) & 0xFF; 

  for (int i = 8; i < 16; i++) // Omron4x4
  {
    bytes[index++] = tdata2[i] & 0xFF;
  }

  for (int i = 0; i < index; i++) // Checksum generation
  {
    checkSum += bytes[i];
  }
  float2Bytes(checkSum, &bytes[index]);
  index += 4;

  bytes[index++] = 0x3E; // >
  bytes[index++] = 0x3E;

  bluetooth.write(bytes, index); // Write bytes to Bluetooth
  bluetooth.print("\n\r");

//  Serial.write(bytes, index); // Write bytes to Bluetooth
//  Serial.print("\n\r");
  ++packet;
  delay(1);


  //  bluetooth.print('@');
  //  SensorData(); // Currently reads both Omron in this function, allowing a single JSON object to be created
  //  OmronData();
  //    OmronData2();

  //  bluetooth.print('\n');
  //  bluetooth.print('%');

}       // End of loop()

////////
// Main functions
///////

void acquireData() {
  DL1 = analogRead(ProxL);
  DS1 = analogRead(ProxS);
  DL2 = analogRead(ProxL2);
  DS2 = analogRead(ProxS2);

  // Adafruit IMU Read
  sensors_event_t accel, mag, gyro, temp;
  lsm.getEvent(&accel, &mag, &gyro, &temp);

  AcX = accel.acceleration.x; // Value in m/s2 
  AcY = accel.acceleration.y;
  AcZ = accel.acceleration.z;

  GyX = gyro.gyro.x; // Value in degrees per second
  GyY = gyro.gyro.y;
  GyZ = gyro.gyro.z;

  // Omron Data
  // Read from Omron 8

  selectMuxPin(omron8_2); // change the addressing
  // Step one - send commands to the sensor
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1); // Delay between instruction and data acquisition
  // Request data from the sensor
  Wire.requestFrom(D6T_addr, 19); // D6T-8 returns 19 bytes
  // Receive the data
  if (0 <= Wire.available()) { // If there is data still left in buffer, we acquire it.
    i = 0;
    for (i = 0; i < 19; i++) {
      rbuf[i] = Wire.read();
    }
    t_PTAT = (rbuf[0] + (rbuf[1] << 8) ) * 0.1;
    //    Serial.print("Omron 8x8,");
//    JsonArray& data = root.createNestedArray("O8-1");
    for (i = 0; i < 8; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8 )) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
//      data.add(tdata[i]);
    }
    //    Serial.print("\n");
  }

  // Read from Omron 4x4 sensor
  selectMuxPin(omron4_2);
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1);
  if (WireExt.beginReception(D6T_addr) >= 0) {
    i = 0;
    // Receive all our bytes of data
    for (i = 0; i < 35; i++) {
      rbuf2[i] = WireExt.get_byte();
    }
    WireExt.endReception(); // End reception
    t_PTAT = (rbuf2[0] + (rbuf2[1] << 8)) * 0.1;
    //    JsonArray& data2 = root.createNestedArray("O16-1");
    // Calculate the individual element values
    for (i = 0; i < 16; i++) {
      tdata2[i] = (rbuf2[(i * 2 + 2)] + (rbuf2[(i * 2 + 3)] << 8)) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");

      //      data2.add(tdata[i]);
    }
  }


}


void SensorData() {
  //  double pitch, roll, yaw;

  // Initialize JSON buffer
  StaticJsonBuffer<500> jsonBuffer; // buffer size 300
  // Create our JSON object
  JsonObject& root = jsonBuffer.createObject();
  int i = 0;

  root["packet"] = packet;
  time_t t = now();
  root["time"] = t;

  //Read from IR sensors
  root["DL1"] = analogRead(ProxL);
  root["DS1"] = analogRead(ProxS);
  root["DL2"] = analogRead(ProxL2);
  root["DS2"] = analogRead(ProxS2);


  // Adafruit IMU Read
  sensors_event_t accel, mag, gyro, temp;
  lsm.getEvent(&accel, &mag, &gyro, &temp);
  //lsm.read();
  //Save IMU Data to JSON String

  root["AcX"] = accel.acceleration.x; // Value in g (Full gravity of 9.81 m/s2 returns a value of 1.0g
  root["AcY"] = accel.acceleration.y;
  root["AcZ"] = accel.acceleration.z;

  root["GyX"] = gyro.gyro.x; // Value in degrees per second
  root["GyY"] = gyro.gyro.y;
  root["GyZ"] = gyro.gyro.z;

  //First Set of Omrons
  // Reading from Omron 8x1 sensor
  selectMuxPin(omron8_2); // change the addressing
  // Step one - send commands to the sensor
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1); // Delay between instruction and data acquisition
  // Request data from the sensor
  Wire.requestFrom(D6T_addr, numbytes); // D6T-8 returns 19 bytes
  // Receive the data
  if (0 <= Wire.available()) { // If there is data still left in buffer, we acquire it.
    i = 0;
    for (i = 0; i < numbytes; i++) {
      rbuf[i] = Wire.read();
    }
    t_PTAT = (rbuf[0] + (rbuf[1] << 8) ) * 0.1;
    //    Serial.print("Omron 8x8,");
    JsonArray& data = root.createNestedArray("O8-1");
    for (i = 0; i < numel; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8 )) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data.add(tdata[i]);
    }
    //    Serial.print("\n");
  }

  // Read from Omron 4x4 sensor
  selectMuxPin(omron4_2);
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1);
  if (WireExt.beginReception(D6T_addr) >= 0) {
    i = 0;
    // Receive all our bytes of data
    for (i = 0; i < 35; i++) {
      rbuf[i] = WireExt.get_byte();
    }
    WireExt.endReception(); // End reception
    // Label our data as Omron 4x4
    //    Serial.print("OMRON 4x4,");
    t_PTAT = (rbuf[0] + (rbuf[1] << 8)) * 0.1;
    JsonArray& data2 = root.createNestedArray("O16-1");
    // Calculate the individual element values
    for (i = 0; i < 16; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8)) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data2.add(tdata[i]);
    }
  }

  root.printTo(bluetooth);
}

//~~~~~~~~~~~~~~~~~

void OmronData() {
  //OMRONS
  // Initialize JSON buffer
  StaticJsonBuffer<500> jsonBuffer; // buffer size 300
  // Create our JSON object
  JsonObject& root = jsonBuffer.createObject();
  int i = 0;

  //First Set of Omrons
  // Reading from Omron 8x1 sensor
  selectMuxPin(omron8); // change the addressing
  // Step one - send commands to the sensor
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1); // Delay between instruction and data acquisition
  // Request data from the sensor
  Wire.requestFrom(D6T_addr, numbytes); // D6T-8 returns 19 bytes
  // Receive the data
  if (0 <= Wire.available()) { // If there is data still left in buffer, we acquire it.
    i = 0;
    for (i = 0; i < numbytes; i++) {
      rbuf[i] = Wire.read();
    }
    t_PTAT = (rbuf[0] + (rbuf[1] << 8) ) * 0.1;
    //    Serial.print("Omron 8x8,");
    JsonArray& data = root.createNestedArray("O8-1");
    for (i = 0; i < numel; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8 )) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data.add(tdata[i]);
    }
    //    Serial.print("\n");
  }

  // Read from Omron 4x4 sensor
  selectMuxPin(omron4);
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1);
  if (WireExt.beginReception(D6T_addr) >= 0) {
    i = 0;
    // Receive all our bytes of data
    for (i = 0; i < 35; i++) {
      rbuf[i] = WireExt.get_byte();
    }
    WireExt.endReception(); // End reception
    // Label our data as Omron 4x4
    //    Serial.print("OMRON 4x4,");
    t_PTAT = (rbuf[0] + (rbuf[1] << 8)) * 0.1;
    JsonArray& data2 = root.createNestedArray("O16-1");
    // Calculate the individual element values
    for (i = 0; i < 16; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8)) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data2.add(tdata[i]);
    }
  }


  //OMRONS DONE

  root.printTo(bluetooth);

}


//~~

void OmronData2() {
  //OMRONS
  // Initialize JSON buffer
  StaticJsonBuffer<500> jsonBuffer; // buffer size 300
  // Create our JSON object
  JsonObject& root = jsonBuffer.createObject();
  int i = 0;

  //Second Set of Omrons
  // Reading from Omron 8x1 sensor
  selectMuxPin(omron8_2); // change the addressing
  // Step one - send commands to the sensor
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1); // Delay between instruction and data acquisition
  // Request data from the sensor
  Wire.requestFrom(D6T_addr, numbytes); // D6T-8 returns 19 bytes
  // Receive the data
  if (0 <= Wire.available()) { // If there is data still left in buffer, we acquire it.
    i = 0;
    for (i = 0; i < numbytes; i++) {
      rbuf[i] = Wire.read();
    }
    t_PTAT = (rbuf[0] + (rbuf[1] << 8) ) * 0.1;
    //    Serial.print("Omron 8x8,");
    JsonArray& data3 = root.createNestedArray("O8-2");
    for (i = 0; i < numel; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8 )) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data3.add(tdata[i]);
    }
    //    Serial.print("\n");
  }

  // Read from Omron 4x4 sensor
  selectMuxPin(omron4_2);
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1);
  if (WireExt.beginReception(D6T_addr) >= 0) {
    i = 0;
    // Receive all our bytes of data
    for (i = 0; i < 35; i++) {
      rbuf[i] = WireExt.get_byte();
    }
    WireExt.endReception(); // End reception
    // Label our data as Omron 4x4
    //    Serial.print("OMRON 4x4,");
    t_PTAT = (rbuf[0] + (rbuf[1] << 8)) * 0.1;
    JsonArray& data4 = root.createNestedArray("O16-2");
    // Calculate the individual element values
    for (i = 0; i < 16; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8)) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data4.add(tdata[i]);
    }
  }

  //OMRONS DONE

  root.printTo(bluetooth);

}

////////
// Other functions
///////


void selectMuxPin(byte pin)
{
  if (pin > 7) return; // Exit if pin is out of scope
  for (int i = 0; i < 3; i++)
  {
    if (pin & (1 << i))
      digitalWrite(selectPins[i], HIGH);
    else
      digitalWrite(selectPins[i], LOW);
  }
}

void float2Bytes(float val, byte* bytes_array)
{
  // Create union of shared memory space
  union {
    float float_variable;
    byte temp_array[4];
  } u;
  // Overite bytes of union with float variable
  u.float_variable = val;
  // Assign bytes to input array
  byte swap_endianness_temp_array[4];
  memcpy(bytes_array, u.temp_array, 4);

}



