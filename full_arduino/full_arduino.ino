//FINAL SKETCH TO RUN MOTORS FROM RASPBERRY/CAMERA COMMANDS
#include <ModbusMaster.h>


// Instantiate ModbusMaster object
ModbusMaster node;
ModbusMaster node1;


// RS485 control pin
#define DE_RE_PIN 3  // Define your DE/RE pin connected to RS485 module (for controlling transmission)
#define DE_DE_PIN 2  // Define your DE/RE pin connected to RS485 module (for controlling transmission)
#define DE_RE_PIN1 7  // Define your DE/RE pin connected to RS485 module (for controlling transmission)
#define DE_DE_PIN1 6  // Define your DE/RE pin connected to RS485 module (for controlling transmission)

void preTransmission() {
  digitalWrite(DE_RE_PIN, HIGH);  // Enable RS485 Transmit
}


void postTransmission() {
  digitalWrite(DE_RE_PIN, LOW);   // Disable RS485 Transmit (set to receive mode)
}

void preTransmission1() {
  digitalWrite(DE_RE_PIN1, HIGH);  // Enable RS485 Transmit
}


void postTransmission1() {
  digitalWrite(DE_RE_PIN1, LOW);   // Disable RS485 Transmit (set to receive mode)
}


void setup() {
  // Begin serial communication at 9600 baud rate (or as specified in your driver documentation)
  Serial.begin(9600);
  Serial2.begin(9600);
  Serial1.begin(9600);

  // Initialize RS485 communication
  pinMode(DE_RE_PIN, OUTPUT);
  digitalWrite(DE_RE_PIN, LOW);  // Set to receive mode initially
  pinMode(DE_RE_PIN1, OUTPUT);
  digitalWrite(DE_RE_PIN1, LOW);  // Set to receive mode initially


  // Initialize Modbus communication with the motor driver
  node.begin(1, Serial);  // 1 = Modbus ID of motor (change this based on your setup)
  node1.begin(1, Serial2);  // 1 = Modbus ID of motor (change this based on your setup)


  // Define pre and post transmission handling functions
  node.preTransmission(preTransmission);
  node.postTransmission(postTransmission);

  node1.preTransmission(preTransmission1);
  node1.postTransmission(postTransmission1);




// Functions for steering on serial 2
void sendStartCommand1() {
  uint16_t controlData = 0x0905;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node1.writeSingleRegister(registerAddress, controlData);
}
void sendStopCommand1() {
  uint16_t controlData = 0x0D05;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node1.writeSingleRegister(registerAddress, controlData);
}
void sendStart2Command1() {
  uint16_t controlData = 0x0B05;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node1.writeSingleRegister(registerAddress, controlData);
}

void sendStallCommand1() {
  uint16_t controlData = 0x0805;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node.writeSingleRegister(registerAddress, controlData);
}


//functions for braking on serial 0
void sendStartCommand() {
  uint16_t controlData = 0x0905;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node.writeSingleRegister(registerAddress, controlData);
}
void sendStopCommand() {
  uint16_t controlData = 0x0D05;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node.writeSingleRegister(registerAddress, controlData);
}
void sendStart2Command() {
  uint16_t controlData = 0x0B05;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node.writeSingleRegister(registerAddress, controlData);
}

void sendStallCommand() {
  uint16_t controlData = 0x0805;  // Example: EN = 1, FR = 1, NW = 1 for RS485 control
  uint16_t registerAddress = 0x8000;  // Address for control register (adjust based on your motor driver)
  uint8_t result = node.writeSingleRegister(registerAddress, controlData);
}
}



void loop() 

{
  // Check for serial input from the Raspberry Pi (now using Serial1 for GPIO TX/RX communication)
  if (Serial1.available() > 0) {
    String command = Serial1.readStringUntil('\n');  // Read the incoming string from Serial1

    // If the command is "START", run the motor
    if (command == "BRAKE") {
      Serial1.println("Received BRAKE command, running motor...");
      sendStartCommand();  // Run motor to pull brakes
      delay(2000);
      sendStopCommand();  // Keep brakes in pulled position  
    }
     if (command == "RELEASE") {
      Serial1.println("Received STOP command, stopping motor...");
      sendStallCommand();  // Release the brakes
    }
  }
}




void loop()
{
if (Serial1.available() > 0) {
    String command = Serial1.readStringUntil('\n');  // Read the incoming string from Serial1

    // If the command is "START", run the motor
    if (command == "CCW") {
      Serial1.println("Steering right");
      sendStart2Command1();  // Steering right
      delay(1500);
      sendStallCommand1();
      
    }
     if (command == "CW") {
      Serial1.println("Steering left");
      sendStartCommand1();  // Steer right
      delay (1500);
    }
    if (command =="STALL"){
      Serial1.println("Cart is alligned straight");
      sendStallCommand1();
    }
  }
