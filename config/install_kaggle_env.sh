#!/usr/bin/env bash

set -e

echo "Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-full

VENV_DIR="$HOME/venvs/kaggle"

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Activating virtual environment and installing Kaggle..."
source "$VENV_DIR/bin/activate"
pip3 install --upgrade pip
pip install kaggle

echo "Creating helper script..."
mkdir -p "$HOME/bin"

cat << 'EOF' > "$HOME/bin/use-kaggle"
#!/usr/bin/env bash
source "$HOME/venvs/kaggle/bin/activate"
EOF

chmod +x "$HOME/bin/use-kaggle"

echo "Setting up Kaggle config directory..."
mkdir -p "$HOME/.kaggle"
chmod 700 "$HOME/.kaggle"

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Move your kaggle.json to ~/.kaggle/"
echo "2. Run: chmod 600 ~/.kaggle/kaggle.json"
echo "3. Activate env: source ~/venvs/kaggle/bin/activate"
echo "4. Test: kaggle datasets list"