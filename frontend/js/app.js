/**
 * Main Application Logic
 * Manages UI state, chat history, and pipeline steps display
 */

// Application State
const AppState = {
    selectedCollection: null,
    selectedModel: null,
    sessionId: null,
    messages: [],
    pipelineSteps: {}, // Map of message index to pipeline steps
    adaptiveN: 1,
};

/**
 * Generate a UUID v4
 */
function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

/**
 * Get or create session ID from localStorage
 */
function getOrCreateSessionId() {
    let sessionId = localStorage.getItem('sessionId');
    if (!sessionId) {
        sessionId = generateUUID();
        localStorage.setItem('sessionId', sessionId);
    }
    return sessionId;
}

/**
 * Create a new session (clear old sessionId and create new one)
 */
function createNewSession() {
    const newSessionId = generateUUID();
    localStorage.setItem('sessionId', newSessionId);
    AppState.sessionId = newSessionId;
    clearChat();
    return newSessionId;
}

/**
 * Initialize the application
 */
async function init() {
    await loadCollections();
    await loadModels();
    
    // Initialize session ID from localStorage or create new one
    AppState.sessionId = getOrCreateSessionId();
    
    // Load state from localStorage if available
    loadStateFromStorage();
    
    // Set up event listeners
    setupEventListeners();
}

/**
 * Load collections from backend
 */
async function loadCollections() {
    try {
        const collections = await CollectionsAPI.list();
        const select = document.getElementById('collection-select');
        select.innerHTML = '<option value="">-- Select Collection --</option>';
        
        collections.forEach(name => {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            if (name === AppState.selectedCollection) {
                option.selected = true;
            }
            select.appendChild(option);
        });
        
        if (AppState.selectedCollection && collections.includes(AppState.selectedCollection)) {
            await loadCollectionInfo(AppState.selectedCollection);
        }
    } catch (error) {
        console.error('Error loading collections:', error);
        // Don't show error if backend is not running - user will see it when they try to use features
        if (error.message.includes('Cannot connect to backend')) {
            const select = document.getElementById('collection-select');
            select.innerHTML = '<option value="">Backend not connected</option>';
        } else {
            showError('Failed to load collections: ' + error.message);
        }
    }
}

/**
 * Load available models
 */
async function loadModels() {
    try {
        const models = await ModelsAPI.list();
        const select = document.getElementById('model-select');
        select.innerHTML = '<option value="">-- Select Model --</option>';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            if (model === AppState.selectedModel) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load models:', error);
        // Silently fail - models selector will just be empty
    }
}

/**
 * Load collection information
 */
async function loadCollectionInfo(collectionName) {
    try {
        const info = await CollectionsAPI.getInfo(collectionName);
        const documents = await CollectionsAPI.getDocuments(collectionName);
        
        // Update UI
        document.getElementById('collection-lang').textContent = (info.language || 'en').toUpperCase();
        document.getElementById('collection-chunks').textContent = info.num_chunks || 0;
        document.getElementById('collection-info').style.display = 'block';
        
        // Display documents
        const documentsList = document.getElementById('documents-list');
        if (documents.total_documents > 0) {
            documentsList.innerHTML = `
                <div class="mb-3 p-3 bg-slate-50 rounded-lg border border-slate-200">
                    <div class="text-xs text-slate-500 mb-1">Total Chunks</div>
                    <div class="text-sm font-semibold text-slate-800">${info.num_chunks || 0}</div>
                </div>
                <div class="mb-2">
                    <div class="text-xs text-slate-500 mb-2">Uploaded Documents (${documents.total_documents})</div>
                    <div class="space-y-1">
                        ${documents.documents.map(doc => `
                            <div class="flex items-center gap-2 p-2 bg-white rounded border border-slate-200 text-xs">
                                <i class="fas fa-file-alt text-indigo-500"></i>
                                <span class="flex-1 text-slate-700">${escapeHtml(doc.filename)}</span>
                                <span class="text-slate-500">${formatFileSize(doc.file_size)}</span>
                                <span class="px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded text-xs font-medium">${doc.num_chunks} chunks</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            document.getElementById('collection-documents').style.display = 'block';
        } else {
            document.getElementById('collection-documents').style.display = 'none';
        }
    } catch (error) {
        showError('Failed to load collection info: ' + error.message);
    }
}

/**
 * Create a new collection
 */
async function createCollection() {
    const name = document.getElementById('new-collection-name').value.trim();
    const language = document.getElementById('new-collection-lang').value;
    
    if (!name) {
        showError('Collection name cannot be empty');
        return;
    }
    
    try {
        showLoading(true);
        await CollectionsAPI.create(name, language);
        document.getElementById('new-collection-name').value = '';
        await loadCollections();
        
        // Auto-select the new collection
        AppState.selectedCollection = name;
        document.getElementById('collection-select').value = name;
        await onCollectionChange();
        
        showSuccess(`Collection '${name}' created successfully`);
    } catch (error) {
        showError('Failed to create collection: ' + error.message);
    } finally {
        showLoading(false);
    }
}

/**
 * Handle collection change
 */
async function onCollectionChange() {
    const select = document.getElementById('collection-select');
    const collectionName = select.value;
    
    if (!collectionName) {
        AppState.selectedCollection = null;
        document.getElementById('collection-info').style.display = 'none';
        clearChat();
        return;
    }
    
    AppState.selectedCollection = collectionName;
    saveStateToStorage();
    
    // Ensure session ID exists (should already be set from init, but double-check)
    if (!AppState.sessionId) {
        AppState.sessionId = getOrCreateSessionId();
    }
    
    await loadCollectionInfo(collectionName);
    clearChat();
}

/**
 * Delete collection
 */
async function deleteCollection() {
    if (!AppState.selectedCollection) {
        return;
    }
    
    if (!confirm(`Are you sure you want to delete collection '${AppState.selectedCollection}'?`)) {
        return;
    }
    
    try {
        showLoading(true);
        await CollectionsAPI.delete(AppState.selectedCollection);
        AppState.selectedCollection = null;
        document.getElementById('collection-select').value = '';
        document.getElementById('collection-info').style.display = 'none';
        clearChat();
        await loadCollections();
        showSuccess('Collection deleted successfully');
    } catch (error) {
        showError('Failed to delete collection: ' + error.message);
    } finally {
        showLoading(false);
    }
}

/**
 * Upload files to collection
 */
async function uploadFiles() {
    if (!AppState.selectedCollection) {
        showError('Please select a collection first');
        return;
    }
    
    const fileInput = document.getElementById('file-upload');
    const files = Array.from(fileInput.files);
    
    if (files.length === 0) {
        showError('Please select at least one file');
        return;
    }
    
    // Log initial state
    console.log(`[UPLOAD UI] Starting upload process`);
    console.log(`[UPLOAD UI] Collection: ${AppState.selectedCollection}`);
    console.log(`[UPLOAD UI] Number of files: ${files.length}`);
    
    // Calculate total file size
    const totalSize = files.reduce((sum, file) => sum + file.size, 0);
    const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
    console.log(`[UPLOAD UI] Total size: ${totalSizeMB} MB`);
    
    try {
        showLoading(true);
        const progressBar = document.getElementById('upload-progress');
        const progressFill = progressBar.querySelector('.progress-fill');
        const progressText = progressBar.querySelector('.progress-text');
        progressBar.style.display = 'block';
        
        // Update progress with status messages
        updateProgress(progressFill, progressText, 10, 'Starting upload...');
        console.log(`[UPLOAD UI] Step 1: Starting upload (10%)`);
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        updateProgress(progressFill, progressText, 20, 'Preparing files...');
        console.log(`[UPLOAD UI] Step 2: Preparing files (20%)`);
        
        // Start the actual upload
        updateProgress(progressFill, progressText, 30, 'Sending files to server...');
        console.log(`[UPLOAD UI] Step 3: Sending files to server (30%)`);
        console.log(`[UPLOAD UI] Calling API at: ${new Date().toISOString()}`);
        
        const uploadStartTime = Date.now();
        
        // Show periodic updates while waiting
        const progressUpdateInterval = setInterval(() => {
            const elapsed = Date.now() - uploadStartTime;
            const elapsedSeconds = Math.floor(elapsed / 1000);
            const elapsedMinutes = Math.floor(elapsedSeconds / 60);
            const remainingSeconds = elapsedSeconds % 60;
            
            let timeText = '';
            if (elapsedMinutes > 0) {
                timeText = `${elapsedMinutes}m ${remainingSeconds}s`;
            } else {
                timeText = `${elapsedSeconds}s`;
            }
            
            updateProgress(progressFill, progressText, 35, `Processing... (${timeText} elapsed)`);
            console.log(`[UPLOAD UI] Still processing... ${timeText} elapsed`);
        }, 5000); // Update every 5 seconds
        
        let result;
        try {
            result = await CollectionsAPI.uploadDocuments(AppState.selectedCollection, files, true);
        } finally {
            clearInterval(progressUpdateInterval);
        }
        
        const uploadTime = Date.now() - uploadStartTime;
        console.log(`[UPLOAD UI] API call completed in ${uploadTime}ms (${(uploadTime / 1000).toFixed(2)}s)`);
        
        updateProgress(progressFill, progressText, 70, 'Processing documents...');
        console.log(`[UPLOAD UI] Step 4: Processing documents (70%)`);
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        updateProgress(progressFill, progressText, 85, 'Refreshing collection info...');
        console.log(`[UPLOAD UI] Step 5: Refreshing collection info (85%)`);
        
        await loadCollectionInfo(AppState.selectedCollection);
        
        updateProgress(progressFill, progressText, 100, 'Completed!');
        console.log(`[UPLOAD UI] Step 6: Completed (100%)`);
        console.log(`[UPLOAD UI] Final result: ${result.num_files} files, ${result.num_chunks} chunks`);
        
        setTimeout(() => {
            progressBar.style.display = 'none';
            if (progressFill) progressFill.style.width = '0%';
            if (progressText) progressText.textContent = '';
        }, 1000);
        
        fileInput.value = '';
        
        const successMsg = `Uploaded ${result.num_files} files, created ${result.num_chunks} chunks`;
        showSuccess(successMsg);
        console.log(`[UPLOAD UI] Success: ${successMsg}`);
    } catch (error) {
        console.error(`[UPLOAD UI] Upload failed:`, error);
        console.error(`[UPLOAD UI] Error stack:`, error.stack);
        showError('Failed to upload files: ' + error.message);
    } finally {
        showLoading(false);
        console.log(`[UPLOAD UI] Upload process finished`);
    }
}

/**
 * Update progress bar and text
 */
function updateProgress(progressFill, progressText, percentage, message) {
    if (progressFill) {
        progressFill.style.width = `${percentage}%`;
    }
    if (progressText) {
        progressText.textContent = message;
    }
    console.log(`[UPLOAD UI] Progress: ${percentage}% - ${message}`);
}

/**
 * Send a chat message
 */
async function sendMessage() {
    if (!AppState.selectedCollection) {
        showError('Please select a collection first');
        return;
    }
    
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    
    if (!query) {
        return;
    }
    
    // Clear input
    input.value = '';
    
    // Add user message to chat
    addMessage('user', query);
    
    // Show loading
    const loadingMessageId = addMessage('assistant', 'Thinking...', true);
    
    try {
        // Get options
        const adaptiveN = parseInt(document.getElementById('adaptive-n').value) || 1;
        const model = document.getElementById('model-select').value || null;
        
        AppState.adaptiveN = adaptiveN;
        AppState.selectedModel = model;
        saveStateToStorage();
        
        // Ensure session ID exists before sending
        if (!AppState.sessionId) {
            AppState.sessionId = getOrCreateSessionId();
        }
        
        // Call API
        const response = await QueryAPI.query(AppState.selectedCollection, query, {
            n: adaptiveN,
            model: model,
            sessionId: AppState.sessionId,
        });
        
        // Update session ID if provided and save to localStorage
        if (response.session_id) {
            AppState.sessionId = response.session_id;
            localStorage.setItem('sessionId', response.session_id);
        }
        
        // Remove loading message
        removeMessage(loadingMessageId);
        
        // Add assistant response
        const messageIndex = addMessage('assistant', response.answer || 'No answer generated.');
        
        // Store pipeline steps
        if (response.pipeline_steps) {
            AppState.pipelineSteps[messageIndex] = response.pipeline_steps;
            updatePipelineStepsDisplay();
        }
        
        // Update query selector
        updateQuerySelector();
        
    } catch (error) {
        removeMessage(loadingMessageId);
        addMessage('assistant', `Error: ${error.message}`);
        showError('Failed to get response: ' + error.message);
    }
}

/**
 * Handle Enter key in chat input
 */
function handleChatInputKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

/**
 * Add a message to the chat
 */
function addMessage(role, content, isLoading = false) {
    const messagesContainer = document.getElementById('chat-messages');
    
    // Remove welcome message if exists
    const welcomeMsg = messagesContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message flex items-start gap-3 mb-4 ${role === 'user' ? 'flex-row-reverse' : ''}`;
    messageDiv.dataset.messageIndex = AppState.messages.length;
    messageDiv.dataset.role = role;
    
    if (isLoading) {
        messageDiv.dataset.loading = 'true';
    }
    
    const avatar = document.createElement('img');
    avatar.className = 'message-avatar w-10 h-10 rounded-full object-cover flex-shrink-0 shadow-sm';
    avatar.src = role === 'user' ? 'assets/user_icon.jpg' : 'assets/bot_icon.jpg';
    avatar.alt = role;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = `message-content max-w-[80%] rounded-2xl px-4 py-3 shadow-sm ${
        role === 'user' 
            ? 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white' 
            : 'bg-white text-slate-800 border border-slate-200'
    }`;
    
    const textDiv = document.createElement('div');
    textDiv.className = 'message-text text-sm leading-relaxed break-words';
    if (isLoading) {
        textDiv.innerHTML = '<div class="flex items-center gap-2"><div class="spinner w-4 h-4 border-2 border-current border-t-transparent rounded-full"></div><span>' + escapeHtml(content) + '</span></div>';
    } else {
        textDiv.innerHTML = formatMarkdown(content);
    }
    
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time text-xs mt-2 opacity-70';
    timeDiv.textContent = new Date().toLocaleTimeString();
    
    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(timeDiv);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Store message
    const messageIndex = AppState.messages.length;
    AppState.messages.push({ role, content, index: messageIndex });
    
    return messageIndex;
}

/**
 * Remove a message from chat
 */
function removeMessage(messageIndex) {
    const messageDiv = document.querySelector(`[data-message-index="${messageIndex}"]`);
    if (messageDiv) {
        messageDiv.remove();
    }
    
    // Remove from state
    AppState.messages = AppState.messages.filter((msg, idx) => idx !== messageIndex);
}

/**
 * Create a new chat session
 */
async function newChat() {
    if (!AppState.selectedCollection) {
        showError('Please select a collection first');
        return;
    }
    
    try {
        // Delete old session from backend if exists
        if (AppState.sessionId) {
            try {
                await SessionsAPI.delete(AppState.sessionId);
            } catch (error) {
                console.warn('Failed to delete old session from backend:', error);
                // Continue anyway - we'll create a new session
            }
        }
        
        // Create new session
        const newSessionId = createNewSession();
        
        // Optionally create session on backend
        try {
            const session = await SessionsAPI.create(AppState.selectedCollection);
            AppState.sessionId = session.session_id;
            localStorage.setItem('sessionId', session.session_id);
        } catch (error) {
            console.warn('Failed to create session on backend:', error);
            // Continue with client-side session ID
        }
        
        clearChat();
        showSuccess('New chat session started');
    } catch (error) {
        showError('Failed to start new chat: ' + error.message);
    }
}

/**
 * Clear chat
 */
function clearChat() {
    const messagesContainer = document.getElementById('chat-messages');
    if (AppState.selectedCollection) {
        messagesContainer.innerHTML = '<div class="welcome-message text-center py-12"><i class="fas fa-comment-dots text-4xl text-slate-400 mb-4"></i><p class="text-slate-600">Start a new conversation. Your previous questions will be remembered for context.</p></div>';
    } else {
        messagesContainer.innerHTML = '<div class="welcome-message text-center py-12"><i class="fas fa-comment-dots text-4xl text-slate-400 mb-4"></i><p class="text-slate-600">Please select or create a collection from the sidebar to start chatting.</p></div>';
    }
    AppState.messages = [];
    AppState.pipelineSteps = {};
    updatePipelineStepsDisplay();
    updateQuerySelector();
}

/**
 * Format markdown text (enhanced implementation)
 */
function formatMarkdown(text) {
    // Escape HTML first
    let html = escapeHtml(text);
    
    // Remove excessive blank lines (more than 1 consecutive newline between paragraphs)
    // Keep only single newlines between paragraphs
    html = html.replace(/\n{2,}/g, '\n');
    
    // Trim leading and trailing whitespace from each line
    const lines = html.split('\n');
    html = lines.map(line => line.trimEnd()).join('\n');
    
    // Process headings (must be done before other processing)
    // Process ### first (h3), then ## (h2), then # (h1) to avoid conflicts
    html = html.replace(/^###\s+(.+)$/gm, '<h3 class="text-base font-semibold mt-3 mb-1.5">$1</h3>');
    html = html.replace(/^##\s+(.+)$/gm, '<h2 class="text-lg font-semibold mt-4 mb-2">$1</h2>');
    html = html.replace(/^#\s+(.+)$/gm, '<h1 class="text-xl font-bold mt-4 mb-2">$1</h1>');
    
    // Bold (must be before italic to avoid conflicts)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>');
    
    // Italic (only single asterisks not part of bold)
    html = html.replace(/(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)/g, '<em class="italic">$1</em>');
    
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code class="bg-black/10 px-1.5 py-0.5 rounded text-xs font-mono">$1</code>');
    
    // Lists - process line by line
    const processedLines = html.split('\n');
    let inList = false;
    let listItems = [];
    let result = [];
    
    for (let i = 0; i < processedLines.length; i++) {
        const line = processedLines[i];
        const trimmedLine = line.trim();
        const listMatch = trimmedLine.match(/^[\-\*]\s+(.+)$/);
        const numberedMatch = trimmedLine.match(/^\d+\.\s+(.+)$/);
        
        // Check if line is empty
        const isEmpty = trimmedLine === '' || trimmedLine === '<br>';
        
        if (listMatch || numberedMatch) {
            if (!inList) {
                inList = true;
                listItems = [];
            }
            listItems.push(listMatch ? listMatch[1] : numberedMatch[1]);
        } else {
            if (inList && listItems.length > 0) {
                const listType = numberedMatch ? 'ol' : 'ul';
                const listClass = numberedMatch ? 'list-decimal' : 'list-disc';
                result.push(`<${listType} class="${listClass} ml-6 my-2">`);
                listItems.forEach(item => {
                    result.push(`<li class="mb-1">${item}</li>`);
                });
                result.push(`</${listType}>`);
                listItems = [];
                inList = false;
            }
            
            // Skip blank lines - we'll handle spacing with single <br> later
            if (!isEmpty) {
                result.push(line);
            }
        }
    }
    
    // Handle list at end of text
    if (inList && listItems.length > 0) {
        result.push('<ul class="list-disc ml-6 my-2">');
        listItems.forEach(item => {
            result.push(`<li class="mb-1">${item}</li>`);
        });
        result.push('</ul>');
    }
    
    html = result.join('\n');
    
    // Convert newlines to <br> (single newline = single <br>)
    html = html.replace(/\n/g, '<br>');
    
    // Clean up: remove <br> tags right after closing tags (headings, lists)
    html = html.replace(/(<\/h[1-3]>|<\/ul>|<\/ol>)\s*<br>/gi, '$1');
    // Remove <br> tags right before opening tags (headings, lists)
    html = html.replace(/<br>\s*(<h[1-3]|<ul|<ol)/gi, '$1');
    // Remove multiple consecutive <br> tags (keep only 1)
    html = html.replace(/(<br>\s*){2,}/gi, '<br>');
    
    return html;
}

/**
 * Update pipeline steps display
 */
function updatePipelineStepsDisplay() {
    const container = document.getElementById('pipeline-steps');
    
    // Find all assistant messages with pipeline steps
    const assistantMessages = [];
    AppState.messages.forEach((msg, idx) => {
        if (msg.role === 'assistant' && AppState.pipelineSteps[idx]) {
            const userQuery = AppState.messages[idx - 1]?.content || '';
            const displayQuery = userQuery.length > 50 ? userQuery.substring(0, 50) + '...' : userQuery;
            assistantMessages.push({
                index: idx,
                label: `Q${assistantMessages.length + 1}: ${displayQuery}`,
                steps: AppState.pipelineSteps[idx],
            });
        }
    });
    
    if (assistantMessages.length === 0) {
        container.innerHTML = '<div class="info-message text-center py-8 text-slate-500 text-sm"><i class="fas fa-info-circle text-2xl mb-3 text-slate-400"></i><p>Pipeline steps will appear here after you ask a question.</p></div>';
        document.getElementById('pipeline-selector').style.display = 'none';
        return;
    }
    
    // Show selector if multiple queries
    if (assistantMessages.length > 1) {
        const selector = document.getElementById('query-select');
        selector.innerHTML = '<option value="">-- Select Query --</option>';
        assistantMessages.forEach(msg => {
            const option = document.createElement('option');
            option.value = msg.index;
            option.textContent = msg.label;
            if (msg.index === assistantMessages[assistantMessages.length - 1].index) {
                option.selected = true;
            }
            selector.appendChild(option);
        });
        document.getElementById('pipeline-selector').style.display = 'block';
        
        // Show latest by default
        const selectedIndex = assistantMessages[assistantMessages.length - 1].index;
        renderPipelineSteps(AppState.pipelineSteps[selectedIndex]);
    } else {
        document.getElementById('pipeline-selector').style.display = 'none';
        renderPipelineSteps(assistantMessages[0].steps);
    }
}

/**
 * Handle query selection change
 */
function onQuerySelect() {
    const select = document.getElementById('query-select');
    const messageIndex = parseInt(select.value);
    
    if (AppState.pipelineSteps[messageIndex]) {
        renderPipelineSteps(AppState.pipelineSteps[messageIndex]);
    }
}

/**
 * Update query selector
 */
function updateQuerySelector() {
    updatePipelineStepsDisplay();
}

/**
 * Render pipeline steps
 */
function renderPipelineSteps(steps) {
    const container = document.getElementById('pipeline-steps');
    
    if (!steps) {
        container.innerHTML = '<div class="info-message text-center py-8 text-slate-500 text-sm"><i class="fas fa-info-circle text-2xl mb-3 text-slate-400"></i><p>No pipeline steps available.</p></div>';
        return;
    }
    
    let html = '<div class="space-y-0">';
    let stepNumber = 0;
    
    // Step 1: Query Rewriting
    if (steps.query_rewriting) {
        stepNumber++;
        const step = steps.query_rewriting;
        html += `
            <div class="pipeline-step-item relative pl-10 pb-6">
                <div class="absolute left-5 top-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-semibold shadow-lg z-10">
                    ${stepNumber}
                </div>
                <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
                    <div class="flex items-center justify-between mb-3 cursor-pointer" onclick="togglePipelineStep(this)">
                        <h4 class="font-semibold text-slate-800 text-sm">Query Rewriting</h4>
                        <span class="px-2.5 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">${(step.time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="pipeline-step-content hidden space-y-2 mt-3">
                        <div class="bg-white rounded-lg p-3 border border-slate-200">
                            <div class="text-xs text-slate-500 mb-1">Original Query</div>
                            <div class="text-sm text-slate-800 font-medium">${escapeHtml(step.original_query || '')}</div>
                        </div>
                        <div class="bg-white rounded-lg p-3 border border-slate-200">
                            <div class="text-xs text-slate-500 mb-1">Rewritten Query</div>
                            <div class="text-sm text-slate-800 font-medium">${escapeHtml(step.rewritten_query || '')}</div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Step 2: Adaptive K Selection
    if (steps.adaptive_k_selection) {
        stepNumber++;
        const step = steps.adaptive_k_selection;
        html += `
            <div class="pipeline-step-item relative pl-10 pb-6">
                <div class="absolute left-5 top-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-semibold shadow-lg z-10">
                    ${stepNumber}
                </div>
                <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
                    <div class="flex items-center justify-between mb-3 cursor-pointer" onclick="togglePipelineStep(this)">
                        <h4 class="font-semibold text-slate-800 text-sm">Adaptive K Selection</h4>
                        <span class="px-2.5 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">${(step.time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="pipeline-step-content hidden space-y-2 mt-3">
                        <div class="grid grid-cols-2 gap-2">
                            <div class="bg-white rounded-lg p-3 border border-slate-200">
                                <div class="text-xs text-slate-500 mb-1">K Selected</div>
                                <div class="text-sm text-slate-800 font-semibold">${step.k_determined || step.k || 0}</div>
                            </div>
                            ${step.average_entropy !== undefined ? `
                                <div class="bg-white rounded-lg p-3 border border-slate-200">
                                    <div class="text-xs text-slate-500 mb-1">Avg Entropy</div>
                                    <div class="text-sm text-slate-800 font-semibold">${step.average_entropy.toFixed(3)}</div>
                                </div>
                            ` : ''}
                            ${step.n ? `
                                <div class="bg-white rounded-lg p-3 border border-slate-200">
                                    <div class="text-xs text-slate-500 mb-1">Iterations</div>
                                    <div class="text-sm text-slate-800 font-semibold">${step.n}</div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Step 3: Retrieval
    if (steps.retrieval) {
        stepNumber++;
        const step = steps.retrieval;
        html += `
            <div class="pipeline-step-item relative pl-10 pb-6">
                <div class="absolute left-5 top-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-semibold shadow-lg z-10">
                    ${stepNumber}
                </div>
                <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
                    <div class="flex items-center justify-between mb-3 cursor-pointer" onclick="togglePipelineStep(this)">
                        <h4 class="font-semibold text-slate-800 text-sm">Retrieval</h4>
                        <span class="px-2.5 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">${(step.time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="pipeline-step-content hidden space-y-3 mt-3">
                        <div class="bg-white rounded-lg p-3 border border-slate-200">
                            <div class="text-xs text-slate-500 mb-1">Found Documents</div>
                            <div class="text-sm text-slate-800 font-semibold">${step.num_results || 0}</div>
                        </div>
                        ${renderRetrievalDetails(step.details)}
                    </div>
                </div>
            </div>
        `;
    }
    
    // Step 4: Answer Generation
    if (steps.answer_generation) {
        stepNumber++;
        const step = steps.answer_generation;
        html += `
            <div class="pipeline-step-item relative pl-10 pb-6">
                <div class="absolute left-5 top-0 w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-xs font-semibold shadow-lg z-10">
                    ${stepNumber}
                </div>
                <div class="bg-slate-50 rounded-xl p-4 border border-slate-200">
                    <div class="flex items-center justify-between mb-3 cursor-pointer" onclick="togglePipelineStep(this)">
                        <h4 class="font-semibold text-slate-800 text-sm">Answer Generation</h4>
                        <span class="px-2.5 py-1 bg-green-100 text-green-700 rounded-full text-xs font-medium">${(step.time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="pipeline-step-content hidden space-y-2 mt-3">
                        <div class="grid grid-cols-2 gap-2">
                            <div class="bg-white rounded-lg p-3 border border-slate-200">
                                <div class="text-xs text-slate-500 mb-1">Model</div>
                                <div class="text-sm text-slate-800 font-semibold">${step.model || 'N/A'}</div>
                            </div>
                            <div class="bg-white rounded-lg p-3 border border-slate-200">
                                <div class="text-xs text-slate-500 mb-1">Chunks Used</div>
                                <div class="text-sm text-slate-800 font-semibold">${step.num_chunks_used || 0}</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Total time
    if (steps.total_time) {
        html += `
            <div class="pl-10">
                <div class="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl p-4 text-white">
                    <div class="flex items-center justify-between">
                        <span class="font-semibold">Total Time</span>
                        <span class="px-3 py-1 bg-white/20 rounded-full text-sm font-medium">${steps.total_time.toFixed(2)}s</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

/**
 * Render retrieval details
 */
function renderRetrievalDetails(details) {
    if (!details) return '';
    
    let html = '<div class="space-y-3">';
    
    // Dense Retrieval
    if (details.dense_retrieval) {
        const dense = details.dense_retrieval;
        html += `
            <div class="bg-white rounded-lg p-3 border border-slate-200">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="text-xs font-semibold text-slate-700">3.1 Dense Retrieval</h5>
                    <span class="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">${(dense.time || 0).toFixed(2)}s</span>
                </div>
                <div class="text-xs text-slate-600 mb-2">
                    Selected candidates: <span class="font-semibold text-slate-800">${dense.num_of_dense_chunk || dense.num_results || 0}</span>
                </div>
                ${renderCandidates(dense.top_10_candidates || [], 'dense')}
            </div>
        `;
    }
    
    // Sparse Retrieval
    if (details.sparse_retrieval) {
        const sparse = details.sparse_retrieval;
        html += `
            <div class="bg-white rounded-lg p-3 border border-slate-200">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="text-xs font-semibold text-slate-700">3.2 Sparse Retrieval</h5>
                    <span class="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">${(sparse.time || 0).toFixed(2)}s</span>
                </div>
                <div class="text-xs text-slate-600 mb-2">
                    Selected candidates: <span class="font-semibold text-slate-800">${sparse.num_of_sparse_chunk || sparse.num_results || 0}</span>
                </div>
                ${renderCandidates(sparse.top_10_candidates || [], 'sparse')}
            </div>
        `;
    }
    
    // Hybrid Retrieval
    if (details.hybrid_retrieval) {
        const hybrid = details.hybrid_retrieval;
        
        // RRF Fusion
        if (hybrid.rrf_rerank) {
            const rrf = hybrid.rrf_rerank;
            const num_passed = rrf.num_results || 0;
            html += `
                <div class="bg-white rounded-lg p-3 border border-slate-200">
                    <div class="flex items-center justify-between mb-2">
                        <h5 class="text-xs font-semibold text-slate-700">3.3.1 RRF Fusion</h5>
                        <span class="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">${(rrf.time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="text-xs text-slate-500 mb-2 p-2 bg-blue-50 rounded border border-blue-100">
                        <strong>RRF prioritizes chunks that are highly ranked by multiple retrieval methods.</strong>
                    </div>
                    <div class="text-xs text-slate-600 mb-2">
                        <strong>Input:</strong> ${rrf.total_candidates_for_rrf || 0} candidates | <strong>Output:</strong> ${num_passed} candidates passed to next stage
                    </div>
                    ${renderRRFTable(rrf.top_10_rrf_candidates || [])}
                </div>
            `;
        }
        
        // Cross-Encoder
        if (hybrid.cross_encoder) {
            const ce = hybrid.cross_encoder;
            html += `
                <div class="bg-white rounded-lg p-3 border border-slate-200">
                    <div class="flex items-center justify-between mb-2">
                        <h5 class="text-xs font-semibold text-slate-700">3.3.2 Cross-Encoder Rerank</h5>
                        <span class="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">${(ce.time || 0).toFixed(2)}s</span>
                    </div>
                    <div class="text-xs text-slate-500 mb-2 p-2 bg-blue-50 rounded border border-blue-100">
                        <strong>Cross-Encoder performs fine-grained relevance scoring on a reduced candidate set for higher accuracy.</strong>
                    </div>
                    <div class="text-xs text-slate-600 mb-2">
                        <strong>Input:</strong> ${ce.selected_candidates?.num_candidates || 0} candidates from RRF | <strong>Applied to:</strong> Small candidate set (top ~20)
                    </div>
                    ${renderCrossEncoderTable(ce.top_10_cross_encoder_candidates || [])}
                </div>
            `;
        }
    }
    
    // Final Top K Chunks
    if (details.output) {
        const output = details.output;
        html += `
            <div class="bg-white rounded-lg p-3 border border-slate-200">
                <div class="flex items-center justify-between mb-2">
                    <h5 class="text-xs font-semibold text-slate-700">3.4 Final Top K Chunks</h5>
                    <span class="px-2 py-0.5 bg-purple-100 text-purple-700 rounded text-xs">${output.num_results || 0} docs</span>
                </div>
                <div class="text-xs text-slate-500 mb-2 p-2 bg-purple-50 rounded border border-purple-100">
                    <strong>Used as final context for answer generation</strong>
                </div>
                ${renderFinalOutputTable(output.documents || [])}
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

/**
 * Render candidates list (kept for backward compatibility with sparse/dense retrieval)
 * Limited to 5 documents with expand/collapse
 */
function renderCandidates(candidates, type) {
    if (!candidates || candidates.length === 0) {
        return '<div class="text-xs text-slate-500 py-2">No candidates available.</div>';
    }
    
    const MAX_VISIBLE = 5;
    const visibleCandidates = candidates.slice(0, MAX_VISIBLE);
    const hiddenCount = candidates.length - MAX_VISIBLE;
    const uniqueId = `candidates-${type}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    let html = '<div class="space-y-2 mt-2">';
    
    // Render visible candidates
    html += visibleCandidates.map((candidate, idx) => {
        const label = candidate.label || 'accepted';
        const score = candidate.rerank_score !== undefined ? candidate.rerank_score : 
                     candidate.rrf_score !== undefined ? candidate.rrf_score : 
                     candidate.score || 0;
        const scoreLabel = candidate.rerank_score !== undefined ? 'Cross-Encoder Score' :
                          candidate.rrf_score !== undefined ? 'RRF Score' :
                          'Score';
        
        const stepPrefix = type === 'final' ? '3.4' : type === 'cross-encoder' ? '3.3.2' : type === 'rrf' ? '3.3.1' : type === 'sparse' ? '3.2' : '3.1';
        
        return `
            <div class="candidate-item bg-slate-50 rounded-lg border border-slate-200 overflow-hidden">
                <div class="candidate-header flex items-center justify-between p-2 cursor-pointer hover:bg-slate-100 transition-colors" onclick="toggleCandidate(this)">
                    <span class="candidate-title text-xs font-medium text-slate-700">
                        ${stepPrefix}.${candidate.index || idx + 1} - ${scoreLabel}: ${score.toFixed(4)}
                        ${label !== 'accepted' ? `<span class="ml-2 px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs">${label}</span>` : ''}
                    </span>
                    <i class="fas fa-chevron-down candidate-score text-xs text-slate-400 transition-transform"></i>
                </div>
                <div class="candidate-content p-3 bg-white border-t border-slate-200">
                    <div class="candidate-meta text-xs text-slate-600 mb-2 space-y-1">
                        <div><strong class="text-slate-700">Source:</strong> ${escapeHtml(candidate.metadata?.source || 'Unknown')}</div>
                        ${candidate.rrf_score !== undefined ? `<div><strong class="text-slate-700">RRF Score:</strong> ${candidate.rrf_score.toFixed(4)}</div>` : ''}
                        ${candidate.rerank_score !== undefined ? `<div><strong class="text-slate-700">Cross-Encoder Score:</strong> ${candidate.rerank_score.toFixed(4)}</div>` : ''}
                    </div>
                    <div class="text-xs text-slate-700 whitespace-pre-wrap leading-relaxed">${escapeHtml(candidate.text || '')}</div>
                </div>
            </div>
        `;
    }).join('');
    
    // Add hidden candidates (collapsed by default)
    if (hiddenCount > 0) {
        html += `<div id="${uniqueId}-hidden" class="hidden space-y-2">`;
        html += candidates.slice(MAX_VISIBLE).map((candidate, idx) => {
            const label = candidate.label || 'accepted';
            const score = candidate.rerank_score !== undefined ? candidate.rerank_score : 
                         candidate.rrf_score !== undefined ? candidate.rrf_score : 
                         candidate.score || 0;
            const scoreLabel = candidate.rerank_score !== undefined ? 'Cross-Encoder Score' :
                              candidate.rrf_score !== undefined ? 'RRF Score' :
                              'Score';
            
            const stepPrefix = type === 'final' ? '3.4' : type === 'cross-encoder' ? '3.3.2' : type === 'rrf' ? '3.3.1' : type === 'sparse' ? '3.2' : '3.1';
            
            return `
                <div class="candidate-item bg-slate-50 rounded-lg border border-slate-200 overflow-hidden">
                    <div class="candidate-header flex items-center justify-between p-2 cursor-pointer hover:bg-slate-100 transition-colors" onclick="toggleCandidate(this)">
                        <span class="candidate-title text-xs font-medium text-slate-700">
                            ${stepPrefix}.${candidate.index || MAX_VISIBLE + idx + 1} - ${scoreLabel}: ${score.toFixed(4)}
                            ${label !== 'accepted' ? `<span class="ml-2 px-1.5 py-0.5 bg-yellow-100 text-yellow-700 rounded text-xs">${label}</span>` : ''}
                        </span>
                        <i class="fas fa-chevron-down candidate-score text-xs text-slate-400 transition-transform"></i>
                    </div>
                    <div class="candidate-content p-3 bg-white border-t border-slate-200">
                        <div class="candidate-meta text-xs text-slate-600 mb-2 space-y-1">
                            <div><strong class="text-slate-700">Source:</strong> ${escapeHtml(candidate.metadata?.source || 'Unknown')}</div>
                            ${candidate.rrf_score !== undefined ? `<div><strong class="text-slate-700">RRF Score:</strong> ${candidate.rrf_score.toFixed(4)}</div>` : ''}
                            ${candidate.rerank_score !== undefined ? `<div><strong class="text-slate-700">Cross-Encoder Score:</strong> ${candidate.rerank_score.toFixed(4)}</div>` : ''}
                        </div>
                        <div class="text-xs text-slate-700 whitespace-pre-wrap leading-relaxed">${escapeHtml(candidate.text || '')}</div>
                    </div>
                </div>
            `;
        }).join('');
        html += '</div>';
        
        // Add toggle button
        html += `
            <button onclick="toggleRetrievalDocuments('${uniqueId}')" 
                    class="w-full mt-2 px-3 py-2 text-xs font-medium text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 rounded-lg border border-indigo-200 transition-colors flex items-center justify-center gap-2">
                <i class="fas fa-chevron-down" id="${uniqueId}-icon"></i>
                <span id="${uniqueId}-text">+ ${hiddenCount} documents hidden</span>
            </button>
        `;
    }
    
    html += '</div>';
    return html;
}

/**
 * Render RRF Fusion table (decision-focused, no text)
 */
function renderRRFTable(candidates) {
    if (!candidates || candidates.length === 0) {
        return '<div class="text-xs text-slate-500 py-2">No candidates available.</div>';
    }
    
    // Find max ranks for color normalization
    let maxSparseRank = 0;
    let maxDenseRank = 0;
    candidates.forEach((candidate) => {
        if (candidate.sparse_rank !== undefined && candidate.sparse_rank !== null) {
            maxSparseRank = Math.max(maxSparseRank, candidate.sparse_rank);
        }
        if (candidate.dense_rank !== undefined && candidate.dense_rank !== null) {
            maxDenseRank = Math.max(maxDenseRank, candidate.dense_rank);
        }
    });
    
    let html = '<div class="overflow-x-auto mt-2">';
    
    // Add explanatory caption
    html += '<div class="text-xs text-slate-600 mb-3 p-2 bg-slate-50 rounded border border-slate-200">';
    html += '<div class="font-medium text-slate-700 mb-1">Each chunk is independently evaluated by:</div>';
    html += '<ul class="list-disc list-inside ml-2 space-y-0.5">';
    html += '<li>Sparse retrieval (keyword matching)</li>';
    html += '<li>Dense retrieval (semantic understanding)</li>';
    html += '</ul>';
    html += '<div class="mt-1.5 text-slate-600">RRF combines these rankings to prioritize chunks supported by multiple perspectives.</div>';
    html += '</div>';
    
    html += '<table class="w-full text-xs border-collapse">';
    html += '<thead><tr class="bg-slate-100 border-b border-slate-200">';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Rank</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Chunk ID</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Source</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">';
    html += '<span class="inline-flex items-center gap-1">';
    html += '<span>🔑</span>';
    html += '<span class="rank-header-tooltip" title="Sparse Rank: ranking based on keyword matching (BM25)">Sparse Rank</span>';
    html += '</span>';
    html += '</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">';
    html += '<span class="inline-flex items-center gap-1">';
    html += '<span>🧠</span>';
    html += '<span class="rank-header-tooltip" title="Dense Rank: ranking based on semantic similarity">Dense Rank</span>';
    html += '</span>';
    html += '</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">RRF Score</th>';
    html += '</tr></thead><tbody>';
    
    candidates.forEach((candidate) => {
        // Calculate color intensity for ranks (lower rank = darker/bolder, higher rank = lighter)
        const sparseRank = candidate.sparse_rank !== undefined && candidate.sparse_rank !== null ? candidate.sparse_rank : null;
        const denseRank = candidate.dense_rank !== undefined && candidate.dense_rank !== null ? candidate.dense_rank : null;
        
        let sparseRankClass = 'text-slate-700';
        let sparseRankWeight = 'font-normal';
        if (sparseRank !== null && maxSparseRank > 0) {
            const normalizedRank = sparseRank / maxSparseRank;
            if (normalizedRank <= 0.2) {
                sparseRankClass = 'text-slate-900';
                sparseRankWeight = 'font-bold';
            } else if (normalizedRank <= 0.4) {
                sparseRankClass = 'text-slate-800';
                sparseRankWeight = 'font-semibold';
            } else if (normalizedRank <= 0.6) {
                sparseRankClass = 'text-slate-700';
                sparseRankWeight = 'font-medium';
            } else {
                sparseRankClass = 'text-slate-500';
                sparseRankWeight = 'font-normal';
            }
        }
        
        let denseRankClass = 'text-slate-700';
        let denseRankWeight = 'font-normal';
        if (denseRank !== null && maxDenseRank > 0) {
            const normalizedRank = denseRank / maxDenseRank;
            if (normalizedRank <= 0.2) {
                denseRankClass = 'text-slate-900';
                denseRankWeight = 'font-bold';
            } else if (normalizedRank <= 0.4) {
                denseRankClass = 'text-slate-800';
                denseRankWeight = 'font-semibold';
            } else if (normalizedRank <= 0.6) {
                denseRankClass = 'text-slate-700';
                denseRankWeight = 'font-medium';
            } else {
                denseRankClass = 'text-slate-500';
                denseRankWeight = 'font-normal';
            }
        }
        
        html += '<tr class="border-b border-slate-100 hover:bg-slate-50">';
        html += `<td class="px-2 py-1.5 text-slate-700">${candidate.index || '-'}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${candidate.chunk_id !== undefined ? candidate.chunk_id : 'N/A'}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${escapeHtml(candidate.metadata?.source || 'Unknown')}</td>`;
        html += `<td class="px-2 py-1.5 ${sparseRankClass} ${sparseRankWeight}">${sparseRank !== null ? sparseRank : '-'}</td>`;
        html += `<td class="px-2 py-1.5 ${denseRankClass} ${denseRankWeight}">${denseRank !== null ? denseRank : '-'}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700 font-mono">${(candidate.rrf_score || 0).toFixed(4)}</td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

/**
 * Render Cross-Encoder Rerank table (decision-focused, no text)
 */
function renderCrossEncoderTable(candidates) {
    if (!candidates || candidates.length === 0) {
        return '<div class="text-xs text-slate-500 py-2">No candidates available.</div>';
    }
    
    let html = '<div class="overflow-x-auto mt-2">';
    
    // Add explanatory caption
    html += '<div class="text-xs text-slate-600 mb-3 p-2 bg-slate-50 rounded border border-slate-200">';
    html += '<div class="font-medium text-slate-700 mb-1">Cross-Encoder jointly evaluates the query and each chunk,</div>';
    html += '<div class="text-slate-600">providing fine-grained relevance scoring on a reduced candidate set.</div>';
    html += '<div class="mt-1.5 text-slate-500 italic">Applied only to top candidates (e.g., top 20) — slower but more precise than Sparse/Dense retrieval.</div>';
    html += '</div>';
    
    html += '<table class="w-full text-xs border-collapse">';
    html += '<thead><tr class="bg-slate-100 border-b border-slate-200">';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Final Rank</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Chunk ID</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Source</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">RRF Rank</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Cross-Encoder Score</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Rank Change</th>';
    html += '</tr></thead><tbody>';
    
    candidates.forEach((candidate) => {
        const rankChange = candidate.rank_change_indicator || '=';
        const rankChangeColor = rankChange === '↑' ? 'text-green-600' : rankChange === '↓' ? 'text-red-600' : 'text-slate-600';
        html += '<tr class="border-b border-slate-100 hover:bg-slate-50">';
        html += `<td class="px-2 py-1.5 text-slate-700">${candidate.final_rank !== undefined ? candidate.final_rank : candidate.index || '-'}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${candidate.chunk_id !== undefined ? candidate.chunk_id : 'N/A'}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${escapeHtml(candidate.metadata?.source || 'Unknown')}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${candidate.rrf_rank !== undefined && candidate.rrf_rank !== null ? candidate.rrf_rank : '-'}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700 font-mono">${(candidate.rerank_score || 0).toFixed(4)}</td>`;
        html += `<td class="px-2 py-1.5 ${rankChangeColor} font-semibold">${rankChange}</td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

/**
 * Render Final Output table (decision-focused, no text)
 * With Accept/Denied status and hover preview
 */
function renderFinalOutputTable(documents) {
    if (!documents || documents.length === 0) {
        return '<div class="text-xs text-slate-500 py-2">No documents available.</div>';
    }
    
    let html = '<div class="overflow-x-auto mt-2"><table class="w-full text-xs border-collapse">';
    html += '<thead><tr class="bg-slate-100 border-b border-slate-200">';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Rank</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Chunk ID</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Source</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Cross-Encoder Score</th>';
    html += '<th class="px-2 py-1.5 text-left font-semibold text-slate-700">Status</th>';
    html += '</tr></thead><tbody>';
    
    documents.forEach((doc, idx) => {
        // Determine status: Accepted by default (all chunks in final output are used)
        // But check if there's a label or status field indicating denied
        const status = doc.label === 'denied' || doc.status === 'denied' ? 'denied' : 'accepted';
        const statusClass = status === 'accepted' ? 'chunk-accepted' : 'chunk-denied';
        const statusIcon = status === 'accepted' ? '✓' : '✕';
        const statusText = status === 'accepted' ? 'Accepted' : 'Denied';
        const statusColor = status === 'accepted' ? 'text-green-700' : 'text-red-700';
        const statusBg = status === 'accepted' ? 'bg-green-50' : 'bg-red-50';
        const statusBorder = status === 'accepted' ? 'border-green-200' : 'border-red-200';
        
        // Create unique ID for hover preview
        const previewId = `preview-${Date.now()}-${idx}`;
        
        // Get preview content (first 200 chars of text) - store raw for later escaping
        const previewTextRaw = doc.text ? doc.text.substring(0, 200) + (doc.text.length > 200 ? '...' : '') : 'No content available';
        const chunkId = doc.chunk_id !== undefined ? String(doc.chunk_id) : 'N/A';
        const source = doc.metadata?.source || 'Unknown';
        
        // Store data in data attributes for safe access
        html += `<tr class="border-b border-slate-100 ${statusClass} chunk-row" 
                     data-preview-id="${previewId}"
                     data-preview-source="${escapeHtml(source)}"
                     data-preview-chunk-id="${escapeHtml(chunkId)}"
                     data-preview-text="${escapeHtml(previewTextRaw)}"
                     onmouseenter="showChunkPreview(event, '${previewId}')"
                     onmouseleave="hideChunkPreview('${previewId}')">`;
        html += `<td class="px-2 py-1.5 text-slate-700">${doc.index || idx + 1}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${chunkId}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700">${escapeHtml(source)}</td>`;
        html += `<td class="px-2 py-1.5 text-slate-700 font-mono">${(doc.rerank_score || 0).toFixed(4)}</td>`;
        html += `<td class="px-2 py-1.5">
                    <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded ${statusBg} ${statusColor} border ${statusBorder} font-medium">
                        <span>${statusIcon}</span>
                        <span>${statusText}</span>
                    </span>
                 </td>`;
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

/**
 * Toggle pipeline step expand/collapse
 */
function togglePipelineStep(header) {
    const step = header.closest('.pipeline-step-item');
    if (step) {
        const content = step.querySelector('.pipeline-step-content');
        if (content) {
            content.classList.toggle('hidden');
        }
    }
}

/**
 * Toggle candidate expand/collapse
 */
function toggleCandidate(header) {
    const item = header.parentElement;
    item.classList.toggle('expanded');
    const icon = header.querySelector('.candidate-score');
    if (icon) {
        icon.classList.toggle('fa-chevron-down');
        icon.classList.toggle('fa-chevron-up');
    }
}

/**
 * Toggle retrieval documents visibility
 */
function toggleRetrievalDocuments(uniqueId) {
    const hiddenDiv = document.getElementById(uniqueId + '-hidden');
    const icon = document.getElementById(uniqueId + '-icon');
    const text = document.getElementById(uniqueId + '-text');
    
    if (!hiddenDiv) return;
    
    const isHidden = hiddenDiv.classList.contains('hidden');
    
    if (isHidden) {
        hiddenDiv.classList.remove('hidden');
        if (icon) {
            icon.classList.remove('fa-chevron-down');
            icon.classList.add('fa-chevron-up');
        }
        if (text) {
            const count = hiddenDiv.querySelectorAll('.candidate-item').length;
            text.textContent = `Hide ${count} documents`;
        }
    } else {
        hiddenDiv.classList.add('hidden');
        if (icon) {
            icon.classList.remove('fa-chevron-up');
            icon.classList.add('fa-chevron-down');
        }
        if (text) {
            const count = hiddenDiv.querySelectorAll('.candidate-item').length;
            text.textContent = `+ ${count} documents hidden`;
        }
    }
}

// Make function available globally
window.toggleRetrievalDocuments = toggleRetrievalDocuments;

/**
 * Show chunk preview on hover
 */
function showChunkPreview(event, previewId) {
    // Get the row element
    const row = event.target.closest('tr');
    if (!row) return;
    
    // Get data from data attributes
    const source = row.getAttribute('data-preview-source') || 'Unknown';
    const chunkId = row.getAttribute('data-preview-chunk-id') || 'N/A';
    const previewText = row.getAttribute('data-preview-text') || 'No content available';
    
    // Remove any existing preview
    const existingPreview = document.getElementById('chunk-preview-tooltip');
    if (existingPreview) {
        existingPreview.remove();
    }
    
    // Create preview tooltip
    const tooltip = document.createElement('div');
    tooltip.id = 'chunk-preview-tooltip';
    tooltip.className = 'chunk-preview-tooltip';
    tooltip.innerHTML = `
        <div class="chunk-preview-header">
            <div class="chunk-preview-title">
                <i class="fas fa-file-alt"></i>
                <strong>${source}</strong>
            </div>
            <div class="chunk-preview-meta">
                <span class="chunk-preview-label">Chunk ID:</span>
                <span class="chunk-preview-value">${chunkId}</span>
            </div>
        </div>
        <div class="chunk-preview-content">
            <div class="chunk-preview-label">Preview:</div>
            <div class="chunk-preview-text">${previewText}</div>
        </div>
    `;
    
    document.body.appendChild(tooltip);
    
    // Position tooltip near the row
    const rect = row.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    
    let left = rect.right + 10;
    let top = rect.top;
    
    // Adjust if tooltip goes off screen
    if (left + tooltipRect.width > window.innerWidth) {
        left = rect.left - tooltipRect.width - 10;
    }
    if (top + tooltipRect.height > window.innerHeight) {
        top = window.innerHeight - tooltipRect.height - 10;
    }
    if (top < 0) {
        top = 10;
    }
    
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
    
    // Fade in animation
    setTimeout(() => {
        tooltip.classList.add('visible');
    }, 10);
}

/**
 * Hide chunk preview
 */
function hideChunkPreview(previewId) {
    const tooltip = document.getElementById('chunk-preview-tooltip');
    if (tooltip) {
        tooltip.classList.remove('visible');
        setTimeout(() => {
            if (tooltip.parentNode) {
                tooltip.remove();
            }
        }, 200);
    }
}

// Make functions available globally
window.showChunkPreview = showChunkPreview;
window.hideChunkPreview = hideChunkPreview;

/**
 * Toggle collapsible section
 */
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    const icon = section.previousElementSibling.querySelector('.toggle-icon');
    section.classList.toggle('active');
    icon.classList.toggle('rotated');
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
}

/**
 * Escape HTML
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Show loading overlay
 */
function showLoading(show) {
    document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}

/**
 * Show error message
 */
function showError(message) {
    alert('Error: ' + message);
}

/**
 * Show success message
 */
function showSuccess(message) {
    // Simple alert for now, can be enhanced with toast notifications
    console.log('Success: ' + message);
}

/**
 * Save state to localStorage
 */
function saveStateToStorage() {
    try {
        localStorage.setItem('ragAppState', JSON.stringify({
            selectedCollection: AppState.selectedCollection,
            selectedModel: AppState.selectedModel,
            adaptiveN: AppState.adaptiveN,
        }));
        // Also save session ID separately
        if (AppState.sessionId) {
            localStorage.setItem('sessionId', AppState.sessionId);
        }
    } catch (error) {
        console.error('Failed to save state:', error);
    }
}

/**
 * Load state from localStorage
 */
function loadStateFromStorage() {
    try {
        const saved = localStorage.getItem('ragAppState');
        if (saved) {
            const state = JSON.parse(saved);
            AppState.selectedCollection = state.selectedCollection || null;
            AppState.selectedModel = state.selectedModel || null;
            AppState.adaptiveN = state.adaptiveN || 1;
            
            if (AppState.adaptiveN) {
                document.getElementById('adaptive-n').value = AppState.adaptiveN;
            }
        }
        // Session ID is loaded separately in init() via getOrCreateSessionId()
    } catch (error) {
        console.error('Failed to load state:', error);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Model selector change
    document.getElementById('model-select').addEventListener('change', (e) => {
        AppState.selectedModel = e.target.value || null;
        saveStateToStorage();
    });
    
    // Adaptive N change
    document.getElementById('adaptive-n').addEventListener('change', (e) => {
        AppState.adaptiveN = parseInt(e.target.value) || 1;
        saveStateToStorage();
    });
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

