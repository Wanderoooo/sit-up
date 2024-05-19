/*
  ESP8266 Blink by Simon Peter
  Blink the blue LED on the ESP-01 module
  This example code is in the public domain

  The blue LED on the ESP-01 module is connected to GPIO1
  (which is also the TXD pin; so we cannot use Serial.print() at the same time)

  Note that this sketch uses LED_BUILTIN to find the pin with the internal LED
*/

#include <ESP8266WiFi.h>        // Include the Wi-Fi library
// #include <ESP8266WiFi.h>
#include <WiFiUdp.h>

const char* ssid     = "hanson-laptop";         // The SSID (name) of the Wi-Fi network you want to connect to
const char* password = "bruhmoment";     // The password of the Wi-Fi network
const char* message = "buzz";
int msg_len = sizeof(message);

#define BUZZER_PIN 16  // Change this to the pin your buzzer is connected to
#define LED_PIN 14
WiFiUDP Udp;
unsigned int localUdpPort = 12345;  // local port to listen on
char incomingPacket[5];  // buffer for incoming packets

void setup() {
  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);         // Start the Serial communication to send messages to the computer
  delay(10);
  Serial.println('\n');
  
  WiFi.begin(ssid, password);             // Connect to the network
  Serial.print("Connecting to ");
  Serial.print(ssid); 
  Serial.println(" ...");

  int i = 0;
  while (WiFi.status() != WL_CONNECTED) { // Wait for the Wi-Fi to connect
    delay(1000);
    Serial.print(++i); Serial.print(' ');
  }

  Serial.println('\n');
  Serial.println("Connection established!");  
  Serial.print("IP address:\t");
  Serial.println(WiFi.localIP());         // Send the IP address of the ESP8266 to the computer
}

void buzz() {
  analogWrite(BUZZER_PIN, 100);  // Send a 50% duty cycle PWM signal to the buzzer
  digitalWrite(LED_PIN, HIGH);
  delay(100);                   // Wait for 1 second
  analogWrite(BUZZER_PIN, 0);    // Turn off the buzzer
  digitalWrite(LED_PIN, LOW);
  delay(80);                   // Wait for 1 second
}

void loop() {
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    // receive incoming UDP packets
    int len = Udp.read(incomingPacket, msg_len);

    if (len > 0) {
      bool allEights = true;

      if (memcmp(incomingPacket, message, msg_len) == 0) {
        buzz();
      }
    }

    Serial.printf("UDP packet contents: %s\n", incomingPacket);
  }
}
