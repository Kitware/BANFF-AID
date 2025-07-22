# BANFF-AID: Banff Automated Nephrology Feature Framework - Artificial Intelligence Diagnosis

BANFF-AID is a plugin for automated nephropathology diagnosis, built on the open-source HistomicsTK platform. It leverages state-of-the-art AI models to streamline and enhance the accuracy of Banff Lesion Score assessments in renal biopsies, addressing key challenges such as precision, automation, and inter-observer variability.

## Getting Started

### Prerequisites
- A working instance of [Digital Slide Archive (DSA)](https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa). The simplest way to install DSA locally is via [Docker Compose](https://github.com/DigitalSlideArchive/digital_slide_archive/tree/master/devops/dsa).
- [Docker](https://docs.docker.com/get-docker/) installed and running on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kitware/BANFF-AID.git
   ```

2. **Build Docker Image:**
   ```bash
   docker build -f banff_aid/Dockerfile -t dzenanz/banff-aid:latest .
   ```

### Integration with DSA

1. Launch your local DSA instance.
2. Import the `dzenanz/banff-aid:latest` Docker image via DSA:
   - Navigate to **Collections → Tasks → Slicer CLI Web Tasks**.
   - Select the CLI import option and import the Docker image.

3. Upload SVS image files and corresponding JSON annotations via the Girder interface (annotation format is subject to upcoming changes).

## Current Capabilities

Phase I of BANFF-AID supports automated Banff Lesion scoring for:
- **Arteriosclerosis**
- **Glomerulosclerosis**
- **Interstitial Fibrosis and Tubular Atrophy (IFTA)**

All outputs and logs are accessible through the job logger within DSA.

## Mission and Goal

Our mission is to reduce clinician workload, enhance precision in diagnostics, and improve efficiency, benefiting clinicians, patients, and healthcare processes. Our primary goal is to contribute to reducing donor kidney discard rates through accurate and reliable assessments.

## References
Please see the full documentation for detailed references and methodology underpinning this work.

