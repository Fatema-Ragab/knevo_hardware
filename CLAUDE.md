# Knevo Exo-Skeleton Device Firmware

This repository contains the embedded firmware for the Knevo exo-skeleton device

## Structure

For historical reasons this repository contains a lot of old code that is no longer relevant but is kept around for reference only.

### Source Code

The only directory of relevance at the moment is

- knevo_stepC_dual_core/

It contains all the code in his current state

### Documentation

Markdown files With additional information

- knevo_firmware_handoff_constraints.md
    - Contains distilled insights collected throughout the journey of Building this firmware which is important to take into consideration while performing any modifications to the code base.
- implementation.md
    - Contains a compact List of what has been implemented
    - knevo_integration_plan.md Is an older planning document that contains more noise but also more information from various stages throughout the implementation.
- phase_2_embedded_gap_analysis.md
    - This contains an analysis of the gap between the contract for the integration with the Mobile application and the current state of the firmware.
- arduino-ide-tools-listing.jpeg
    - The tools listing from inside the Arduino IDE

## Libraries and dependencies

### TFLiteMicro_ArduinoESP32S3

<https://github.com/j-siderius/TFLiteMicro_ArduinoESP32S3>

>Arduino library that ports TensorFlow Lite Micro to the ESP32-S3. Provides a very basic API for abstracting some of the TFLite c-specific references and nuances.
