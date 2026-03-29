#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$ROOT/data/raw/nyc_taxi"
ZONE_DIR="$RAW_DIR/zones"

mkdir -p "$RAW_DIR" "$ZONE_DIR"

echo "=== NYC TLC Taxi Zone Files ==="
if [ ! -f "$ZONE_DIR/taxi_zones.zip" ]; then
    curl -L -o "$ZONE_DIR/taxi_zones.zip" \
        https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
fi
# Extract zip if shapefile not found (zip contains taxi_zones/taxi_zones.shp)
if [ ! -f "$ZONE_DIR/taxi_zones/taxi_zones.shp" ]; then
    echo "  Extracting taxi_zones.zip ..."
    unzip -o "$ZONE_DIR/taxi_zones.zip" -d "$ZONE_DIR/"
fi

if [ ! -f "$ZONE_DIR/taxi_zone_lookup.csv" ]; then
    curl -L -o "$ZONE_DIR/taxi_zone_lookup.csv" \
        https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv
else
    echo "  taxi_zone_lookup.csv already exists, skipping"
fi

echo ""
echo "=== NYC TLC Yellow Taxi Parquet (2024, all 12 months) ==="
BASE="https://d37ci6vzurychx.cloudfront.net/trip-data"
for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    fname="yellow_tripdata_2024-${month}.parquet"
    if [ ! -f "$RAW_DIR/$fname" ]; then
        echo "  Downloading $fname ..."
        curl -L -o "$RAW_DIR/$fname" "${BASE}/$fname"
    else
        echo "  $fname already exists, skipping"
    fi
done

echo ""
echo "=== Overture Maps ==="
OVERTURE="$ROOT/data/raw/gowalla/overture_us_places.parquet"
if [ -f "$OVERTURE" ]; then
    echo "  Reusing existing Overture parquet: $OVERTURE"
else
    echo "  WARNING: Overture US places not found at $OVERTURE"
    echo "  Run gowalla_enrich_overture.py first, or download manually."
fi

echo ""
echo "Done. Files:"
ls -lh "$RAW_DIR"/*.parquet 2>/dev/null || echo "  (no parquet files yet)"
ls -lh "$ZONE_DIR"/ 2>/dev/null || true
