#if ! (ESP8266 || ESP32 )
  #error This code is intended to run on the ESP8266/ESP32 platform! Please check your Tools->Board setting
#endif

char ssid[] = "6517(1)_2.4GHz";             // your network SSID (name)
char pass[] = "Cse6517!";         // your network password

char user[]         = "user";          // MySQL user login username
char password[]     = "1234";          // MySQL user login password

#define MYSQL_DEBUG_PORT      Serial

// Debug Level from 0 to 4
#define _MYSQL_LOGLEVEL_      1
#include <Arduino.h>
#include <MySQL_Generic.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <BLEAdvertisedDevice.h>
#include <BLEEddystoneURL.h>
#include <BLEEddystoneTLM.h>
#include <BLEBeacon.h>

#define USING_HOST_NAME     false

IPAddress SQL_server(10, 125, 68, 60);

#define ENDIAN_CHANGE_U16(x) ((((x)&0xFF00) >> 8) + (((x)&0xFF) << 8))

int scanTime = 1;
BLEScan *pBLEScan;
char * mybeacon = "ESP32";

uint16_t server_port = 3306; 

const int id = 1;

char default_database[] = "esp32";
char default_table[]    = "item";

MySQL_Connection conn((Client *)&client);
MySQL_Query *query_mem;
MySQL_Query sql_query = MySQL_Query(&conn);
float value = 0;

SimpleKalmanFilter kf(2, 2, 0.01);

class MyAdvertisedDeviceCallbacks : public BLEAdvertisedDeviceCallbacks{
    void onResult(BLEAdvertisedDevice advertisedDevice){
      if(strcmp(advertisedDevice.getName().c_str(),mybeacon) == 0){
        int ble_rssi = advertisedDevice.getRSSI();
        value = kf.updateEstimate(ble_rssi);
        Serial.print(value);
      }
    }
};

void setup()
{
  Serial.begin(115200);
  
  //beacon setup
  BLEDevice::init("");
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setActiveScan(true);
  pBLEScan->setInterval(10);
  pBLEScan->setWindow(9);
  
  //query setup
  MYSQL_DISPLAY1("\nStarting Basic_Insert_ESP on", ARDUINO_BOARD);
  MYSQL_DISPLAY(MYSQL_MARIADB_GENERIC_VERSION);
  // Begin WiFi section
  MYSQL_DISPLAY1("Connecting to", ssid);  
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED){
    delay(500);
    MYSQL_DISPLAY0(".");
  }
  // print out info about the connection:
  //MYSQL_DISPLAY1("Connected to network. My IP address is:", WiFi.localIP());
  //MYSQL_DISPLAY3("Connecting to SQL Server @", SQL_server, ", Port =", server_port);
  //MYSQL_DISPLAY5("User =", user, ", PW =", password, ", DB =", default_database);
  
}

int count = 0;
String query = "";
String INSERT_SQL = "";

void loop()
{ 
  BLEScanResults foundDevices = pBLEScan->start(scanTime, false);
  INSERT_SQL = String("INSERT INTO ") + "esp32.item(num_,value_,count_) VALUES ('1','" + value + "', '" + count + "')";
  count++;
  pBLEScan->clearResults();
  
  if (conn.connectNonBlocking(SQL_server, server_port, user, password) != RESULT_FAIL && value != 0)
  {
    runInsert();
    conn.close();                     
  } 
  else 
  {
    MYSQL_DISPLAY("\nConnect failed or zero value");
  }
}

void runInsert()
{
  MySQL_Query query_mem = MySQL_Query(&conn);

  if (conn.connected())
  {
    MYSQL_DISPLAY(INSERT_SQL);
    
    if ( !query_mem.execute(INSERT_SQL.c_str()) )
    {
      MYSQL_DISPLAY("Insert error");
    }
    else
    {
      MYSQL_DISPLAY("Data Inserted.");
    }
  }
  else
  {
    MYSQL_DISPLAY("Disconnected from Server. Can't insert.");
  }
}
