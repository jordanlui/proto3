/* Script reads data from Prototype device and outputs to serial in JSON Format
  Sensors include Omron, Sharp IR sensor, IMU, and FSRs.
  Multiplexer setup is used to access the OMRON
  This script reads the data from a 8x1 OMRON D6T-8L sensor using only the Wire library.
  Note that use of D6T-44L 4x4 sensor requires the WireExt.h library, so that the 35 bytes of data can be read from the sensor.
  Original code from Arduino forum on WireExt.h https://forum.arduino.cc/index.php?topic=217394.0


  Pin Assignments

  c1 Omron 8x1 on mux
  c2 omron 4x4 on mux
  A0, A1 for Sharp IR Sensors
  A2, A3, for FSRs

  Reminder on Multiplexing:
  Multiplexer tutorial - http://bildr.org/2011/02/cd74hc4067-arduino/
  Multiplex the SDA signals to respect cx pins. Read mux pin 0 with Arduino.
  All i2c component SCL lines connect to regular Arduino SCL line.
  Note Arduino Uno i2c pin arrangements. A4 (SDA), A5 (SCL)
  Adding in the FSRs.


*/

#include <Wire.h>
#include <WireExt.h>
#include <elapsedMillis.h>
#include <ArduinoJson.h>
#include <SPI.h>

// IMU Setup
#include <Adafruit_LSM9DS0.h>
#include <Adafruit_Sensor.h>
Adafruit_LSM9DS0 lsm = Adafruit_LSM9DS0();
void setupSensor()
{
  // 1.) Initialize each component
  lsm.setupAccel(lsm.LSM9DS0_ACCELRANGE_2G);
  lsm.setupMag(lsm.LSM9DS0_MAGGAIN_2GAUSS);
  lsm.setupGyro(lsm.LSM9DS0_GYROSCALE_245DPS);

}
// End of IMU Setup

//elapsedMillis timeElapsed; // Global variable declaration for the elapsed time in the script as variable timeElapsed. Time output is not working yet :(

// Omron Sensor setup
#define D6T_addr 0x0A // 7 bit address of OMRON D6T is 0x0A in hex, 0000 1010 in binary
#define D6T_cmd 0x4C // Standard command is 4C in hex, 0100 1100 in binary
int numbytes = 19;
int numel = 8;
int rbuf[19]; // Actual raw data is coming in as 35 bytes for OMRON 4x4, and 19 bytes for Omron 8x1.
int tdata[8]; // The data comming from omron8 is in 8 elements
int rbuf2[35]; // Actual raw data is coming in as 35 bytes for OMRON 4x4
int tdata2[16]; // The data comming from the omron4x4 is in 16 elements
float t_PTAT;

// Think this code is from the old IMU. TO BE DELETED
//#define Gyro_addr 0x6B

// Multiplexer setup
//Mux control pins
int s0 = 8;
int s1 = 9;
int s2 = 10;
int s3 = 11;
//Mux in "SIG" pin. The pin we actually receive and send signal to
int SIG_pin = 0;
int muxdelay = 10; // Delay time in ms that we insert after mux pin changes

// Pin setups
// FSR pins read together since we need to conserve analog pins
int fsr1 = 2; // nominal elbow extension
int fsr2 = 2; // nominal elbow flexion
int fsr3 = 2; // affected elbow extension
int fsr4 = 3; // affected elbow flexion
int fsr5 = 3; // finger fsr for measurinig upward elbow flexion, supination
int fsr6 = 3; // fsr that we might not use

// Digital pins are used to selectively power the FSRs, effectively multiplexing our system
int dfsr1 = 7; // nominal elbow extension
int dfsr2 = 8; // nominal elbow flexion
int dfsr3 = 9; // affected elbow extension
int dfsr4 = 7; // affected elbow flexion
int dfsr5 = 8; // finger fsr for measurinig upward elbow flexion, supination
int dfsr6 = 9; // fsr that we might not use

// Omron mux pins
int omron8 = 1; // mux address
int omron4 = 2; // mux address

// Sharp Sensor setup
int ProxL = 0; // The long range IR sensor, 10-80 cm. Analog addres
int ProxS = 1; // The short range IR sensor, 2-15 cm. Analog address

void setup()
{

  // Adafruit IMU Setup
  // We read IMU right from the i2c main line - no mux required.
#ifndef ESP8266
  while (!Serial);     // will pause Zero, Leonardo, etc until serial console opens
#endif
  Serial.begin(9600);
  //  Serial.println("LSM raw read demo");
  // Try to initialise and warn if we couldn't detect the chip
  if (!lsm.begin())
  {
    Serial.println("Oops ... unable to initialize the LSM9DS0. Check your wiring!");
    while (1);
  }
  Serial.println("LSM9DS0 9DOF IMU Detected.");
  Serial.println("");

  // Multiplex setup
  pinMode(s0, OUTPUT);
  pinMode(s1, OUTPUT);
  pinMode(s2, OUTPUT);
  pinMode(s3, OUTPUT);
  digitalWrite(s0, LOW);
  digitalWrite(s1, LOW);
  digitalWrite(s2, LOW);
  digitalWrite(s3, LOW);

  // Digital pin setup for FSR power multiplexing
  pinMode(dfsr1, OUTPUT);
  digitalWrite(dfsr1, LOW);
  pinMode(dfsr2, OUTPUT);
  digitalWrite(dfsr2, LOW);
  pinMode(dfsr3, OUTPUT);
  digitalWrite(dfsr3, LOW);

  Wire.begin();
  //Serial.begin(9600);

}

void loop()
{

  // Initialize JSON buffer
  StaticJsonBuffer<350> jsonBuffer; // buffer size 300
  // Create our JSON object
  JsonObject& root = jsonBuffer.createObject();
  int i = 0;

  //Read from IR sensors
  root["dist1"] = analogRead(ProxL);
  root["dist2"] = analogRead(ProxS);

  // Reading from Omron 8x1 sensor
  readMux(omron8); // change the addressing
  // Step one - send commands to the sensor
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(20); // Delay between instruction and data acquisition
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
    JsonArray& data = root.createNestedArray("omron8");
    for (i = 0; i < numel; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8 )) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data.add(tdata[i]);
    }
    //    Serial.print("\n");
  }

  // Read from Omron 4x4 sensor
  readMux(omron4);
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(20);
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
    JsonArray& data2 = root.createNestedArray("omron16");
    // Calculate the individual element values
    for (i = 0; i < 16; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8)) * 0.1;
      //      Serial.print(tdata[i]);
      //      Serial.print(",");
      data2.add(tdata[i]);
    }
  }



  // Read FSRs
  root["fsr1"] = readFSR(fsr1, dfsr1);
  root["fsr2"] = readFSR(fsr2, dfsr2);
  root["fsr3"] = readFSR(fsr3, dfsr3);
  root["fsr4"] = readFSR(fsr4, dfsr4);
  root["fsr5"] = readFSR(fsr5, dfsr5);
  root["fsr6"] = readFSR(fsr6, dfsr6);

  // Adafruit IMU Read
  lsm.read();
  // Save IMU Data to JSON String
  root["orientation1"] = (int)lsm.gyroData.x;
  root["orientation2"] = (int)lsm.gyroData.y;
  root["orientation3"] = (int)lsm.gyroData.z;
  // Print out the IMU data (for debug)
  //  Serial.print("Accel X: "); Serial.print((int)lsm.accelData.x); Serial.print(" ");
  //  Serial.print("Y: "); Serial.print((int)lsm.accelData.y);       Serial.print(" ");
  //  Serial.print("Z: "); Serial.println((int)lsm.accelData.z);     Serial.print(" ");
  //  Serial.print("Mag X: "); Serial.print((int)lsm.magData.x);     Serial.print(" ");
  //  Serial.print("Y: "); Serial.print((int)lsm.magData.y);         Serial.print(" ");
  //  Serial.print("Z: "); Serial.println((int)lsm.magData.z);       Serial.print(" ");
  //  Serial.print("Gyro X: "); Serial.print((int)lsm.gyroData.x);   Serial.print(" ");
  //  Serial.print("Y: "); Serial.print((int)lsm.gyroData.y);        Serial.print(" ");
  //  Serial.print("Z: "); Serial.println((int)lsm.gyroData.z);      Serial.println(" ");

  // Print our output objects
  // root.prettyPrintTo(Serial); // pretty print is an option
  root.printTo(Serial);
  Serial.println();

  delay(10); // Main loop delay
}

////////
// Other functions
///////

int readFSR(int pin, int dpin) {
  // Read the FSRs at the specified analog pin. Note a digital pin must be specified to power the FSR for the sampling period.
  int result;

  // Turn on voltage to the select FSR
  digitalWrite(dpin, HIGH);

  // Read from the analog pin
  result = analogRead(pin);
  digitalWrite(dpin, LOW);


  return result;

}

int readMux(int channel) {
  // Changes the addressing of the MUX.
  int controlPin[] = {s0, s1, s2, s3};

  int muxChannel[16][4] = {

    {0, 0, 0, 0}, //channel 0
    {1, 0, 0, 0}, //channel 1
    {0, 1, 0, 0}, //channel 2
    {1, 1, 0, 0}, //channel 3
    {0, 0, 1, 0}, //channel 4
    {1, 0, 1, 0}, //channel 5
    {0, 1, 1, 0}, //channel 6
    {1, 1, 1, 0}, //channel 7
    {0, 0, 0, 1}, //channel 8
    {1, 0, 0, 1}, //channel 9
    {0, 1, 0, 1}, //channel 10
    {1, 1, 0, 1}, //channel 11
    {0, 0, 1, 1}, //channel 12
    {1, 0, 1, 1}, //channel 13
    {0, 1, 1, 1}, //channel 14
    {1, 1, 1, 1} //channel 15
  };

  //Write these changes to the mux channel addresses
  for (int i = 0; i < 4; i ++) {

    digitalWrite(controlPin[i], muxChannel[channel][i]);
  }

  delay(muxdelay);
}
