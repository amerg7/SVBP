<br />
<p align="center">
  <a href="https://seu.edu.sa/en/home/">
    <img src="https://seu.edu.sa/media/1387/logo_3d_last.png" alt="Logo" width="300" height="90">
  </a>

  <h3 align="center">Smart Vision for Blind People(SVBP)</h3>
  <p align="center">
    <br />
    <a href="https://youtu.be/TPsLbUpEoQY">View Demo</a>
    ·
    <a href="https://github.com/amerg7/SVBP/issues/new">Report Bug</a>
  </p>

<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#how-to-start">How to start</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#project-partners">Project Partners</a></li>
  </ol>
</details>


## About The Project
The project is a smart eyeglass named “Smart Vision for Blind People” (SVBP) for people with complete or partial
blindness by giving them information about their surrounding environment including the people by using face/object
detection and recognition and obstacle avoidance. The SVBP system will output voice-based guidance to guide the person
through his/her path and read the environment using a camera that takes inputs from the environment and its contents,
calculate the distance between the camera’s location and all near objects, based on that calculation, SVBP will give
information about all objects such as names, distance, and in case of a living object, it will give the name of the
person, an approximate distance separating the user from the person, an approximate indication of the user’s facial
expression. All this information generated from SVBP will have a form of physical output which is through the use of
headphones/speakers.

### Built With

* []() Python3


## Getting Started

### Prerequisites

For running Python Recommended System Requirements

* Processors:Intel® Core™ i5 processor 4300M at 2.60 GHz or 2.59 GHz (1 socket, 2 cores, 2 threads per core), 
  8 GB of DRAMIntel® Xeon® processor E5-2698 v3 at 2.30 GHz (2 sockets, 16 cores each, 1 thread per core), 
  64 GB of DRAMIntel® Xeon Phi™ processor 7210 at 1.30 GHz (1 socket, 64 cores, 4 threads per core), 
  32 GB of DRAM, 16 GB of MCDRAM (flat mode enabled)
* Disk space: 2 to 3 GB
* Operating systems: Windows® 10, macOS*, and Linux*, Raspbian*.

Minimum System Requirements:
* Processors: Intel Atom® processor or Intel® Core™ i3 processor
* Disk space: 1 GB
* Operating systems: Windows* 7 or later, macOS, and Linux, Raspbian*.

For running Tensorflow Minimum Systems Requirements:
* Python 3.6–3.8
* Ubuntu 16.04 or later
* Windows 7 or later (with C++ redistributable)
* macOS 10.12.6 (Sierra) or later (no GPU support)
* Raspbian 9.0 or later


### Installation

To use the code you must first install all the libraries needed below.

* []() OpenCv 
  ```sh
  pip install opencv-python
  ```
* []() OpenCv contrib
  ```sh
  pip install opencv-contrib-python
  ```
* []() TensorFlow - You need to install C++ redistributable before installing TensorFlow.
  ```sh
  pip install tensorflow
  ```
* []() Keras
  ```sh
  pip install Keras
  ```
* []() Pickle
  ```sh
  pip install pickle-mixin
  ```
* []() PIL
  ```sh
  pip install Pillow
  ```
* []() Numpy - installed by Defult with Python3
   ```sh
  installed by Defult with Python3
  ```
* []() Shapely
   ```sh
  pip install Shapely
  ```
* []() Math
  ```sh
  pip install python-math
  ```
* []() Pyttsx3
  ```sh
  pip install pyttsx3
  ```
  
## How to start

When you open the program with any IDE, run the `SmartVision_GUI.py`file.

  <a href="https://raw.githubusercontent.com/amerg7/SVBP/main/SVBP-GUI.png">
<img src="https://raw.githubusercontent.com/amerg7/SVBP/main/SVBP-GUI.png" alt="SVBP-GUI" width="106" height="223">
  </a>

   * Put your name and give yourself an ID number then press Save.
   * Face the camera then press Take Picture.
   * Wait until the program finishes training your image.
   * Press start Smart vision.
   * enjoy


## Roadmap
See the [open issues](https://github.com/amerg7/SVBP/issues) for a list of proposed features (and known issues).

## Project Partners

* Amer(Me): [https://github.com/amerg7](https://github.com/amerg7)
* Naji: [https://github.com/OLD-GOLD](https://github.com/OLD-GOLD)
* Naif:  [https://github.com/Naccie](https://github.com/Naccie)


