# whatsapp-chat-analyzer
Tool to get insights into a whatsapp chat.

![showcase](showcase.png){width="600"}


## Usage
1. Clone the repository and install the required dependencies.
   ```bash
   git clone https://github.com/yourusername/whatsapp-chat-analyzer.git
   cd whatsapp-chat-analyzer
   pip install -r requirements.txt
   ```
2. Run the report generator with the path to your WhatsApp chat export file.
   ```bash
   python report.py --file path/to/whatsapp_chat.txt
   ```

Note: Since some visualizations require plotly you may need to install a chrome instance using:
```bash
plotly_get_chrome
```
Note: You might need to install 
```bash 
sudo apt install libcairo2-dev pkg-config python3-dev
```
