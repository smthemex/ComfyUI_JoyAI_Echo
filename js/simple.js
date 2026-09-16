const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "JoyAIEchoExtension",
    
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeType.comfyClass !== "JoyAI_Echo_SM_Encoder") return;
        if (nodeData && nodeData.input && nodeData.input.required && nodeData.input.required.prompt_files) {
            nodeData.input.required.prompt_files = ["JSON_UPLOAD", {}];
        }
    },

    getCustomWidgets() {
        return {
            JSON_UPLOAD(node, name) {
                const button = document.createElement("button");
                button.textContent = "Select json file upload";
                button.style.cssText = `
                    margin-top: 10px;
                    padding: 0 10px;
                    background: #222222;
                    border: 1px solid #d3d3d3;
                    border-radius: 4px;
                    color: white;
                    cursor: pointer;
                    width: 100%;
                    height: 30px;
                    line-height: 30px;
                    text-align: center;
                    text-overflow: ellipsis;
                    overflow: hidden;
                    white-space: nowrap;
                    box-sizing: border-box;
                `;

                const input = document.createElement('input');
                input.type = 'file';
                input.style.display = 'none';
                input.accept = '.json';
                input.multiple = false;
                input.nwdirectory = false;

                input.onchange = async e => {
                    const file = e.target.files[0];
                    if (!file) return;
                    const fileName = file.name;

                    const reader = new FileReader();
                    reader.onload = async (event) => {
                        try {
                            const fileContent = event.target.result.split(',')[1];
                            const response = await fetch('/joyai_echo/get_file_path', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    filename: fileName,
                                    content: fileContent,
                                    node_id: node.id
                                })
                            });
                            const data = await response.json();
                            console.log('API response:', data);

                            const w = node.widgets.find(w => w.name === name);
                            if (w) {
                                w.value = data.path;
                                if (w.callback) w.callback(data.path);
                            }
                            button.textContent = fileName;
                        } catch (error) {
                            console.error('Error sending file to backend:', error);
                        }
                    };
                    reader.readAsDataURL(file);
                };

                button.onclick = () => input.click();

                const widget = node.addDOMWidget(name, "json_upload", button, {
                    serialize: false,
                    hideOnZoom: false
                });
                widget.serialize = false;
                return { widget, value: "" };
            }
        };
    }
});
