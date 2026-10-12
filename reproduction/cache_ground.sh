#!/usr/bin/env bash
# Preserve NVIDIA's exact ground asset on clusters without direct S3 access.
# Run at repository root with https_proxy set when required.
set -euo pipefail
source_url=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid
destination=outputs/nvidia-assets/Isaac/Environments/Grid
mkdir -p "$destination/Materials/Textures"
for asset in default_environment.usd Materials/Textures/Wireframe_blue.png Materials/Textures/WireframeBlur_basecolor.png Materials/Textures/WireframeBlur_blue.png; do
  curl --fail --location --max-time 60 "$source_url/$asset" -o "$destination/$asset"
done
echo "8a21c317d638d33a4e6c20c958a7ed8f4d8c5195efd7e20f1b85178b4a15638a  $destination/default_environment.usd" | sha256sum --check
sha256sum "$destination/default_environment.usd" "$destination"/Materials/Textures/*.png > outputs/nvidia-ground-sha256.txt
