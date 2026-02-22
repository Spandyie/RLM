#!/bin/bash

# RLM Document Chatbot - Start Script
# This script starts both the backend and frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   RLM Document Chatbot Launcher${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Ollama is running
echo -e "\n${YELLOW}Checking Ollama...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
else
    echo -e "${RED}✗ Ollama is not running${NC}"
    echo -e "${YELLOW}Starting Ollama...${NC}"
    ollama serve &
    sleep 3
fi

# Check if llama3.1:8b is available
echo -e "\n${YELLOW}Checking for llama3.1:8b model...${NC}"
if ollama list | grep -q "llama3.1:8b"; then
    echo -e "${GREEN}✓ llama3.1:8b model is available${NC}"
else
    echo -e "${YELLOW}Pulling llama3.1:8b model (this may take a while)...${NC}"
    ollama pull llama3.1:8b
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Start backend in background
echo -e "\n${YELLOW}Starting FastAPI backend on port 8000...${NC}"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2

# Check if backend started successfully
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend is running at http://localhost:8000${NC}"
else
    echo -e "${RED}✗ Backend failed to start${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo -e "\n${YELLOW}Starting Streamlit frontend on port 8501...${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Open http://localhost:8501 in your browser${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Handle cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null
    echo -e "${GREEN}Done!${NC}"
}
trap cleanup EXIT

# Start Streamlit (foreground)
streamlit run frontend/app.py --server.port 8501
