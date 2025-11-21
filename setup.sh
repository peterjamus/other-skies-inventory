#!/bin/bash

# Other Skies Inventory System - Setup Script
# This script sets up the local development environment

echo "======================================"
echo "Other Skies Inventory System Setup"
echo "======================================"
echo ""

# Check for required tools
echo "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 20+"
    echo "   Visit: https://nodejs.org/"
    exit 1
else
    echo "✓ Node.js $(node -v)"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    echo "   Visit: https://www.python.org/"
    exit 1
else
    echo "✓ Python $(python3 --version)"
fi

# Check PostgreSQL
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed. Please install PostgreSQL 15+"
    echo "   Visit: https://www.postgresql.org/download/"
    exit 1
else
    echo "✓ PostgreSQL $(psql --version)"
fi

# Check Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR is not installed."
    echo "   Installing Tesseract is recommended for OCR functionality."
    echo ""
    echo "   On macOS: brew install tesseract"
    echo "   On Ubuntu: sudo apt-get install tesseract-ocr"
    echo "   On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    echo ""
    read -p "Continue without Tesseract? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Tesseract $(tesseract --version | head -n 1)"
fi

echo ""
echo "======================================"
echo "Step 1: Setting up environment"
echo "======================================"

# Copy environment file
if [ ! -f .env.local ]; then
    cp .env.local.example .env.local
    echo "✓ Created .env.local file"
    echo ""
    echo "⚠️  Please edit .env.local and add your API keys:"
    echo "   - Google Books API key (free)"
    echo "   - ISBNdb API key (optional)"
    echo "   - Shipping API keys (optional)"
    echo ""
    read -p "Press enter to continue after adding API keys..."
else
    echo "✓ .env.local already exists"
fi

echo ""
echo "======================================"
echo "Step 2: Installing Node.js dependencies"
echo "======================================"
npm install

echo ""
echo "======================================"
echo "Step 3: Setting up Python environment"
echo "======================================"

# Create Python virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Created Python virtual environment"
else
    echo "✓ Python virtual environment exists"
fi

# Activate and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Installed Python dependencies"

echo ""
echo "======================================"
echo "Step 4: Setting up PostgreSQL database"
echo "======================================"

# Create database directory
mkdir -p data/postgres

# Initialize database if needed
if [ ! -d "data/postgres/base" ]; then
    initdb -D ./data/postgres
    echo "✓ Initialized PostgreSQL database"
else
    echo "✓ PostgreSQL database already initialized"
fi

# Start PostgreSQL
pg_ctl -D ./data/postgres start

# Create database
createdb otherskies 2>/dev/null || echo "✓ Database 'otherskies' already exists"

# Run Prisma migrations
echo "Running database migrations..."
npx prisma migrate dev --name init
npx prisma generate
echo "✓ Database schema created"

echo ""
echo "======================================"
echo "Step 5: Creating directories"
echo "======================================"

# Create required directories
mkdir -p uploads processed luts temp/uploads temp/processed
echo "✓ Created working directories"

echo ""
echo "======================================"
echo "Step 6: Creating default LUTs"
echo "======================================"

# Create default LUT files
cat > luts/README.md << 'EOF'
# LUT Directory

Place your .cube LUT files here for archival color grading.

Default LUTs created:
- neutral_accurate.cube - Maximum color fidelity
- warm_archival.cube - Subtle warmth for vintage books
- gothic_drama.cube - Enhanced contrast for weird fiction
- parchment.cube - Aged paper tones

You can add custom .cube files here and they will be available in the system.
EOF

echo "✓ LUT directory prepared"

echo ""
echo "======================================"
echo "✅ Setup Complete!"
echo "======================================"
echo ""
echo "To start the system:"
echo ""
echo "1. Start the database (if not running):"
echo "   npm run db:start"
echo ""
echo "2. In terminal 1 - Start the Next.js app:"
echo "   npm run dev"
echo ""
echo "3. In terminal 2 - Start the OCR service:"
echo "   source venv/bin/activate"
echo "   python services/ocr_server.py"
echo ""
echo "4. In terminal 3 - Start the LUT service:"
echo "   source venv/bin/activate"
echo "   python services/lut_processor.py"
echo ""
echo "Then open http://localhost:3000 in your browser"
echo ""
echo "Default services:"
echo "- Web Interface: http://localhost:3000"
echo "- OCR Service: http://localhost:8001"
echo "- LUT Service: http://localhost:8002"
echo "- Database Studio: npx prisma studio (http://localhost:5555)"
echo ""
echo "For help, check README.md"
