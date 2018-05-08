// Code for proto 6
// Device acquires Omron and IMU data and transmits over bluetooth. IMU data acquired at higher speed, omron data repeated where necessary

#include "quaternionFilters.h"
#include "MPU9250.h"
#include <Wire.h>
#include <SoftwareSerial.h>
#include <WireExt.h>

MPU9250 myIMU;

// Omron Sensor setup~~~~
#define D6T_addr 0x0A // 7 bit address of OMRON D6T is 0x0A in hex, 0000 1010 in binary
#define D6T_cmd 0x4C // Standard command is 4C in hex, 0100 1100 in binary

int rbuf[35]; // Actual raw data is coming in as 35 bytes for OMRON 4x4, and 19 bytes for Omron 8x1.
unsigned int tdata[16]; // The data comming from omron8 is in 8 elements and 16 elements for omron4x4
unsigned char tdatashift[16];
unsigned int t_PTAT;
int i = 0;

// IMU setup
unsigned int acceldata[3];
unsigned int gyrodata[3];
float quat[4];
unsigned int quatshift[4];

// Byte setup
unsigned char temphighbyte[16];
unsigned char templowbyte[16];

unsigned char accelhighbyte[3];
unsigned char accellowbyte[3];

unsigned char gyrohighbyte[3];
unsigned char gyrolowbyte[3];

unsigned char quathighbyte[4];
unsigned char quatlowbyte[4];


//Bluetooth setup
int bluetoothTx = 2;  // TX-O pin of bluetooth mate, Arduino D2
int bluetoothRx = 3;  // RX-I pin of bluetooth mate, Arduino D3



SoftwareSerial bluetooth(bluetoothTx, bluetoothRx);


void setup()
{

  Wire.begin();
  Serial.begin(115200);
  bluetooth.begin(115200);  // The Bluetooth Mate defaults to 115200bps
  delay(100);  // Short delay, wait for the Mate to send back CMD 
  myIMU.initMPU9250();
  myIMU.initAK8963(myIMU.magCalibration);
}

void loop()
{
  //Variable output setup
  int sendomron = 5; // every # iteration will send omron instead of IMU
  int outputcount = 0;
  
  // Time stamp
  unsigned int deltatime = 0;

  while (1)
  {
      outputcount++;
      if(outputcount == sendomron)
      {
        readomron();   
        getbytes();

        for (i = 0; i < 16; i++)
        {
          bluetooth.write(tdatashift[i]);  
        }
        
        bluetooth.write('\n');
         
        outputcount = 0;
      }

      else
      {
        
        IMUdata();    
        getbytes();

        for (i = 0; i < 3; i++)
        {
          bluetooth.write(accelhighbyte[i]);
          bluetooth.write(accellowbyte[i]);
        }
  
        for (i = 0; i < 3; i++)
        {
          bluetooth.write(gyrohighbyte[i]);
          bluetooth.write(gyrolowbyte[i]);
        }
  
        for (i = 0; i < 4; i++)
        {
          bluetooth.write(quathighbyte[i]);
          bluetooth.write(quatlowbyte[i]);
        } 
        deltatime = millis();
        bluetooth.write(highByte(deltatime));
        bluetooth.write(lowByte(deltatime));
                
        bluetooth.write('\n');
      }
  
  }
}


void readomron()
{
  
  Wire.beginTransmission(D6T_addr);
  Wire.write(D6T_cmd);
  Wire.endTransmission();
  delay(1);
  if (WireExt.beginReception(D6T_addr) >= 0)
  {
    i = 0;
    // Receive all our bytes of data
    for (i = 0; i < 35; i++)
    {
      rbuf[i] = WireExt.get_byte();
    }
    WireExt.endReception(); // End reception
    t_PTAT = (rbuf[0] + (rbuf[1] << 8));
    //    JsonArray& data2 = root.createNestedArray("O16-1");
    // Calculate the individual element values
    for (i = 0; i < 16; i++) {
      tdata[i] = (rbuf[(i * 2 + 2)] + (rbuf[(i * 2 + 3)] << 8));

    }
  }
  
  for (int i = 0; i < 16; i++) // limits the range of omron values from 405-150
  {
    if (tdata[i] > 405) // 405 max temp
    {
      tdatashift[i]=405-150;
    }
    else if(tdata[i]<150) // 150 min temp
    {
      tdatashift[i]=0;
    }
    else
    {
    tdatashift[i]=tdata[i]-150;
    
    }

  }
}


void IMUdata()
{
 
  //if (myIMU.readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01)
  //{
    // Read the x/y/z adc values
    myIMU.readAccelData(myIMU.accelCount);
    myIMU.getAres();
    // Now we'll calculate the accleration value into actual g's
    // This depends on scale being set
    myIMU.ax = ((float)myIMU.accelCount[0] * myIMU.aRes)-0.14; // - accelBias[0];
    myIMU.ay = ((float)myIMU.accelCount[1] * myIMU.aRes)-0.06; // - accelBias[1];
    myIMU.az = ((float)myIMU.accelCount[2] * myIMU.aRes)+0.17; // - accelBias[2];

    acceldata[0] =  (myIMU.ax) * 100 + 800;
    acceldata[1] =  (myIMU.ay) * 100 + 800;
    acceldata[2] =  (myIMU.az) * 100 + 800;

    // Read the x/y/z adc values
    myIMU.readGyroData(myIMU.gyroCount);
    myIMU.getGres();
    // Calculate the gyro value into actual degrees per second
    // This depends on scale being set
    myIMU.gx = ((float)myIMU.gyroCount[0] * myIMU.gRes)+1;
    myIMU.gy = ((float)myIMU.gyroCount[1] * myIMU.gRes)-2;
    myIMU.gz = ((float)myIMU.gyroCount[2] * myIMU.gRes)-0.3;

    gyrodata[0] =  (myIMU.gx) * 10 + 32500;
    gyrodata[1] =  (myIMU.gy) * 10 + 32500;
    gyrodata[2] =  (myIMU.gz) * 10 + 32500;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    myIMU.readMagData(myIMU.magCount);  // Read the x/y/z adc values
    myIMU.getMres();
    // User environmental x-axis correction in milliGauss, should be
    // automatically calculatedde
    myIMU.magBias[0] = +470.;
    // User environmental x-axis correction in milliGauss TODO axis??
    myIMU.magBias[1] = +120.;
    // User environmental x-axis correction in milliGauss
    myIMU.magBias[2] = +125.;

    // Calculate the magnetometer values in milliGauss
    // Include factory calibration per data sheet and user environmental
    // corrections
    // Get actual magnetometer value, this depends on scale being set
    myIMU.mx = (float)myIMU.magCount[0] * myIMU.mRes * myIMU.magCalibration[0] -
               myIMU.magBias[0];
    myIMU.my = (float)myIMU.magCount[1] * myIMU.mRes * myIMU.magCalibration[1] -
               myIMU.magBias[1];
    myIMU.mz = (float)myIMU.magCount[2] * myIMU.mRes * myIMU.magCalibration[2] -
               myIMU.magBias[2];

  myIMU.updateTime();
  MahonyQuaternionUpdate(myIMU.ax, myIMU.ay, myIMU.az, myIMU.gx * DEG_TO_RAD,
                         myIMU.gy * DEG_TO_RAD, myIMU.gz * DEG_TO_RAD, myIMU.my,
                         myIMU.mx, myIMU.mz, myIMU.deltat);

  quat[0] = *getQ();
  quat[1] = *(getQ() + 1);
  quat[2] = *(getQ() + 2);
  quat[3] = *(getQ() + 3);
  quatshift[0] = quat[0] * 100 + 800;
  quatshift[1] = quat[1] * 100 + 800;
  quatshift[2] = quat[2] * 100 + 800;
  quatshift[3] = quat[3] * 100 + 800;
}


void getbytes()
{
  

  for (i = 0; i < 16; i++)
  {
    
    if(tdatashift[i]==0xA)
    {
      tdatashift[i]=0xB;
    }
  }
  
  for (i = 0; i < 3; i++)
  {
    accelhighbyte[i] = highByte(acceldata[i]);
    accellowbyte[i] = lowByte(acceldata[i]);
  
  if(accelhighbyte[i]==0xA)
    {
      accelhighbyte[i]=0xB;
    }
    if(accelhighbyte[i]==0xA)
    {
      accellowbyte[i]=0xB;
    }
  }
  
  for (i = 0; i < 3; i++)
  {
    gyrohighbyte[i] = highByte(gyrodata[i]);
    gyrolowbyte[i] = lowByte(gyrodata[i]);

    if(gyrohighbyte[i]==0xA)
    {
      gyrohighbyte[i]=0xB;
    }
    if(gyrohighbyte[i]==0xA)
    {
      gyrolowbyte[i]=0xB;
    }
  }
  
  for (i = 0; i < 4; i++)
  {
    quathighbyte[i] = highByte(quatshift[i]);
    quatlowbyte[i] = lowByte(quatshift[i]);

    if(quathighbyte[i]==0xA)
    {
      quathighbyte[i]=0xB;
    }
    if(quathighbyte[i]==0xA)
    {
      quatlowbyte[i]=0xB;
    }
    
  }
}

