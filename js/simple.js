const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "JoyAIEchoExtension",
    nodeCreated(node) {
        if (node.comfyClass === "JoyAI_Echo_SM_Encoder") {
            // 创建按钮
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
            
            /// 创建文件选择输入
            const input = document.createElement('input');
            input.type = 'file';
            input.style.display = 'none';
            
            input.accept = '.json';
           
            input.multiple = false;
      
            input.nwdirectory = false;

            input.onchange = async e => {
                const file = e.target.files[0];
                if (file) {
                    const fileName = file.name;
                    //button.textContent = fileName;
                    
                    const reader = new FileReader();
                    reader.onload = async (event) => {
                        try {
                            const fileContent = event.target.result.split(',')[1];
                            
                            const response = await fetch('/joyai_echo/get_file_path', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                    filename: fileName,
                                    content: fileContent,
                                    node_id: node.id
                                })
                            });
                            
                            const data = await response.json();
                            console.log('API response:', data);
                            
                            // 将后端返回的绝对路径写入到节点的 prompt_files 控件中
                            const promptFilesWidget = node.widgets.find(w => w.name === 'prompt_files');
                            if (promptFilesWidget) {
                                promptFilesWidget.value = data.path;
                                // 触发回调，确保ComfyUI内部状态更新（如标记工作流为未保存）
                                if (promptFilesWidget.callback) {
                                    promptFilesWidget.callback(data.path);
                                }
                            }
                        } catch (error) {
                            console.error('Error sending file to backend:', error);
                        }
                    };
                    reader.readAsDataURL(file);
                }
            };

            
            // 按钮点击事件触发文件选择
            button.onclick = () => {
                input.click();
            };
            
            // 添加DOM widget
            node.addDOMWidget("path-button", "button", button, {
                serialize: false,
                hideOnZoom: false
            });
            
            // 初始添加一个输入框用于存储路径
            node.addInput('selected_path', 'STRING', {
                default: '',
                multiline: false,
                tooltip: 'Selected file path'
            });
        }
    }
});
