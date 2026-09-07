// Copyright 2025 TESOLLO
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the TESOLLO nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace DeltoTCP {

// ============================================================================
// Model IDs
// ============================================================================
enum class ModelType : uint16_t {
  DG1F      = 0x1F02,   // DG-1F
  DG2F      = 0x2F02,   // DG-2F-M
  DG3F_B    = 0x3F01,   // DG-3F B (구형, 5바이트/모터)
  DG3F_M    = 0x3F02,   // DG-3F M (신형, 8바이트/모터)
  DG4F      = 0x4F02,   // DG-4F
  DG5F      = 0x5F02,   // DG-5F
  DG5F_L    = 0x5F12,   // DG-5F LEFT
  DG5F_R    = 0x5F22,   // DG-5F RIGHT
  DG5F_L_S  = 0x5F14,   // DG-5F LEFT S
  DG5F_R_S  = 0x5F24,   // DG-5F RIGHT S
  DG5F_L_S15 = 0x5F34,  // DG-5F LEFT S15
  DG5F_R_S15 = 0x5F44,  // DG-5F RIGHT S15
};

enum class DataCode: uint8_t {
  JOINT      = 0x01,
  CURRENT    = 0x02,
  TEMPERATURE= 0x03,
  VELOCITY   = 0x04,
  FINGERTIP_SENSOR = 0x05,
  GPIO       = 0x06,
  MODULE_ERROR = 0x07,
};


enum class SensorType : uint8_t {
  NONE      = 0x00,
  FT_6AXIS  = 0x01,   // 6 Axis Force/Torque, 12byte/finger
  FT_3AXIS  = 0x02,   // 3 Axis (x,y,z, nc, nc, nc),  12byte/finger
  TACTILE_M = 0x03,   // 3x5, 1byte/cell, 15byte/finger
  FT_4AXIS  = 0x04,   // 4 Axis (x,y,z, nc, nc, tz), 12byte/finger
  TACTILE_S = 0x05,   // 3x6, 2byte/cell, 36byte/finger
};

// ============================================================================
// Data Structures
// ============================================================================
struct DeltoReceivedData {
  std::vector<double> joint;             // radian
  std::vector<double> current;           // A
  std::vector<double> temperature;       // Celsius
  std::vector<double> velocity;          // rad/s
  std::vector<bool> gpio;                // GPIO states (4 elements)
  std::vector<double> fingertip_sensor;  // F/T sensor data (N fingers × 6 axes)
  std::vector<std::vector<uint8_t>> tactile_m;   // Tactile M: per finger, 3x5=15 bytes (uint8)
  std::vector<std::vector<uint16_t>> tactile_s;  // Tactile S: per finger, 3x6=18 values (uint16)
};

// ============================================================================
// Communication Class
// ============================================================================
class Communication {
 public:
  Communication(const std::string& ip, int port, uint16_t model,
                bool fingertip_sensor = false, bool io = false);
  ~Communication();

  // Connection management
  void Connect();
  void Disconnect();
  bool IsConnected() const { return sockfd_ >= 0; }

  // Data exchange
  DeltoReceivedData GetData();
  void SendDuty(std::vector<int>& duty);

  // GPIO control
  void SetGPIO(bool output1, bool output2, bool output3);

  // F/T sensor control
  void SetFTSensorOffset();

  // Version info
  std::vector<uint8_t> GetFirmwareVersion();

  // Model info
  uint16_t GetActualModel() const { return actual_model_; }
  uint16_t GetConfiguredModel() const { return model_; }
  static std::string ModelToString(uint16_t model);

  // Sensor info
  SensorType GetSensorType() const { return sensor_type_; }
  uint8_t GetFingerSensorMask() const { return finger_sensor_mask_; }
  int GetSensorFingerCount() const;

 private:
  // Connection parameters
  std::string ip_;
  int port_;
  uint16_t model_;
  uint16_t actual_model_;
  bool fingertip_sensor_;
  bool io_;
  SensorType sensor_type_;
  uint8_t finger_sensor_mask_;  // bit i = finger i+1 has a sensor

  // Network (POSIX socket)
  int sockfd_;
  std::vector<uint8_t> firmware_version_;

  // Model-specific parameters
  const int motor_count_;
  const int byte_per_motor_;
  const int finger_count_;
  int expected_response_length_;  // int, not int16_t: must not wrap negative

  // Protocol constants
  static constexpr uint8_t GET_DATA_CMD = 0x01;
  static constexpr uint8_t SET_DUTY_CMD = 0x05;
  static constexpr uint8_t SET_GPIO_CMD = 0x06;
  static constexpr uint8_t GET_VERSION_CMD = 0x08;
  static constexpr uint8_t SET_FT_SENSOR_OFFSET_CMD = 0x0B;

  // length (2 bytes) + command (1 byte) + payload (variable)
  static constexpr std::size_t HEADER_SIZE = 3;

  // Sanity cap for the device-declared response length
  static constexpr std::size_t MAX_RESPONSE_LENGTH = 1024;

  // 0.1 deg to radian: 0.1 * (pi/180) = pi/1800
  // S Model 0.01 deg to radian 0.01 * (pi/180) = pi/18000
  static constexpr double POSITION_SCALE = (M_PI / 1800.0);
  static constexpr double POSITION_SCALE_S = (M_PI / 18000.0);
  
  // mA to mA (no scaling, but keep for clarity and future adjustments)
  static constexpr double CURRENT_SCALE = 1.0;

  // 1rpm to rad/s: (2 * pi) / 60 = pi / 30, 
  static constexpr double VELOCITY_SCALE = (M_PI / 30.0);
  
  static constexpr int GPIO_SIZE = 4;

  // Helper functions
  int GetMotorCount(uint16_t model);
  int GetBytePerMotor(uint16_t model);
  int GetFingerCount(uint16_t model);
  int GetSensorBytesPerFinger() const;
  int CalculateExpectedResponseLength();
  bool SendAll(const uint8_t* data, std::size_t len, int timeout_ms = 500);
  bool RecvAll(uint8_t* data, std::size_t len, int timeout_ms = 500);
  bool ReadFullPacket(std::vector<uint8_t>& buffer);
  void DrainSocket();  // drop stale bytes so a late reply is not mis-read
  int16_t CombineMsg(uint8_t data1, uint8_t data2);
  int8_t ConvertByte(uint8_t byte);
  bool IsNewModel() const;
  bool SupportsExtendedFeatures() const;
};

}  // namespace DeltoTCP
