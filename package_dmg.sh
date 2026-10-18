#!/bin/bash
APP_NAME="Depth Map v1"
DMG_NAME="${APP_NAME}.dmg"
STAGING_DIR="dmg_staging"

# Clean up
rm -rf "$STAGING_DIR"
rm -f "$DMG_NAME"
mkdir -p "$STAGING_DIR"

# Copy app to staging
cp -R "${APP_NAME}.app" "$STAGING_DIR/"

# Create link to Applications
ln -s /Applications "$STAGING_DIR/Applications"

# Create DMG
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGING_DIR" -ov -format UDZO "$DMG_NAME"

# Clean up
rm -rf "$STAGING_DIR"
echo "DMG created: $DMG_NAME"
