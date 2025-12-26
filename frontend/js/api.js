/**
 * API Client for Backend FastAPI
 * Handles all HTTP requests to the backend server
 */

const API_BASE_URL = 'http://localhost:2022';

/**
 * Generic fetch wrapper with error handling
 */
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const config = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        
        // Provide more helpful error messages
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            throw new Error('Cannot connect to backend. Please ensure the backend server is running on http://localhost:2022');
        }
        
        throw error;
    }
}

/**
 * Collections API
 */
const CollectionsAPI = {
    /**
     * List all collections
     */
    async list() {
        return await apiRequest('/collections');
    },
    
    /**
     * Create a new collection
     */
    async create(name, language = 'en') {
        return await apiRequest('/collections', {
            method: 'POST',
            body: JSON.stringify({ name, language }),
        });
    },
    
    /**
     * Delete a collection
     */
    async delete(name) {
        return await apiRequest(`/collections/${name}`, {
            method: 'DELETE',
        });
    },
    
    /**
     * Get collection info
     */
    async getInfo(name) {
        return await apiRequest(`/collections/${name}/info`);
    },
    
    /**
     * Get collection documents
     */
    async getDocuments(name) {
        return await apiRequest(`/collections/${name}/documents`);
    },
    
    /**
     * Get collection chunks
     */
    async getChunks(name, skip = 0, limit = 100) {
        return await apiRequest(`/collections/${name}/chunks?skip=${skip}&limit=${limit}`);
    },
    
    /**
     * Upload documents to a collection
     */
    async uploadDocuments(name, files, useSemanticChunking = true) {
        const startTime = Date.now();
        console.log(`[UPLOAD] Starting upload to collection: ${name}`);
        console.log(`[UPLOAD] Files to upload: ${files.length}`);
        
        // Log file details
        files.forEach((file, index) => {
            const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
            console.log(`[UPLOAD] File ${index + 1}: ${file.name} (${fileSizeMB} MB, type: ${file.type || 'unknown'})`);
        });
        
        console.log(`[UPLOAD] Use semantic chunking: ${useSemanticChunking}`);
        
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        // Match Streamlit: send as string "true" or "false" to match backend expectation
        formData.append('use_semantic_chunking', useSemanticChunking ? 'true' : 'false');
        
        const url = `${API_BASE_URL}/collections/${name}/documents`;
        console.log(`[UPLOAD] Sending request to: ${url}`);
        console.log(`[UPLOAD] Request started at: ${new Date().toISOString()}`);
        
        // Calculate timeout based on file size (1 minute per MB, minimum 5 minutes, maximum 30 minutes)
        const totalSizeMB = files.reduce((sum, f) => sum + f.size, 0) / (1024 * 1024);
        const timeoutMs = Math.max(5 * 60 * 1000, Math.min(30 * 60 * 1000, totalSizeMB * 60 * 1000));
        console.log(`[UPLOAD] Request timeout set to: ${(timeoutMs / 1000 / 60).toFixed(1)} minutes (${(timeoutMs / 1000).toFixed(0)}s)`);
        
        // Create AbortController for timeout
        const abortController = new AbortController();
        const timeoutId = setTimeout(() => {
            abortController.abort();
            console.error(`[UPLOAD] Request timeout after ${(timeoutMs / 1000).toFixed(0)}s`);
        }, timeoutMs);
        
        // Periodic logging to show request is still pending
        const logInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const elapsedSeconds = (elapsed / 1000).toFixed(0);
            const elapsedMinutes = (elapsed / 60000).toFixed(1);
            console.log(`[UPLOAD] Request still pending... (${elapsedSeconds}s / ${elapsedMinutes}min elapsed)`);
            console.log(`[UPLOAD] Backend is processing - this may take a while for large files with semantic chunking`);
        }, 10000); // Log every 10 seconds
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
                signal: abortController.signal,
            });
            
            // Clear timeout and interval on success
            clearTimeout(timeoutId);
            clearInterval(logInterval);
            
            const requestTime = Date.now() - startTime;
            console.log(`[UPLOAD] Response received after ${requestTime}ms (${(requestTime / 1000).toFixed(2)}s)`);
            console.log(`[UPLOAD] Response status: ${response.status} ${response.statusText}`);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                console.error(`[UPLOAD] Error response:`, errorData);
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            const totalTime = Date.now() - startTime;
            console.log(`[UPLOAD] Upload completed successfully in ${totalTime}ms (${(totalTime / 1000).toFixed(2)}s)`);
            console.log(`[UPLOAD] Result:`, result);
            console.log(`[UPLOAD] Files uploaded: ${result.num_files}, Chunks created: ${result.num_chunks}`);
            
            return result;
        } catch (error) {
            // Clear timeout and interval on error
            clearTimeout(timeoutId);
            clearInterval(logInterval);
            
            const totalTime = Date.now() - startTime;
            console.error(`[UPLOAD] Upload failed after ${totalTime}ms (${(totalTime / 1000).toFixed(2)}s)`);
            console.error(`[UPLOAD] Error:`, error);
            
            if (error.name === 'AbortError') {
                const errorMsg = `Request timeout after ${(timeoutMs / 1000 / 60).toFixed(1)} minutes. The file may be too large or backend is taking too long to process.`;
                console.error(`[UPLOAD] ${errorMsg}`);
                throw new Error(errorMsg);
            } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                console.error(`[UPLOAD] Network error - backend may be down or unreachable`);
                throw new Error('Cannot connect to backend. Please ensure the backend server is running on http://localhost:2022');
            } else {
                throw error;
            }
        }
    },
};

/**
 * Query API
 */
const QueryAPI = {
    /**
     * Execute a query on a collection
     */
    async query(collectionName, query, options = {}) {
        const payload = {
            query,
            n: options.n || null,
            model: options.model || null,
            session_id: options.sessionId || null,
            language: options.language || null,
        };
        
        // Remove null values
        Object.keys(payload).forEach(key => {
            if (payload[key] === null) {
                delete payload[key];
            }
        });
        
        return await apiRequest(`/collections/${collectionName}/query`, {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
};

/**
 * Models API
 */
const ModelsAPI = {
    /**
     * Get available models
     */
    async list() {
        return await apiRequest('/models');
    },
};

/**
 * Sessions API
 */
const SessionsAPI = {
    /**
     * Create a new session
     */
    async create(collectionName = null) {
        const payload = collectionName ? { collection_name: collectionName } : {};
        return await apiRequest('/sessions', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
    
    /**
     * Delete a session
     */
    async delete(sessionId) {
        return await apiRequest(`/sessions/${sessionId}`, {
            method: 'DELETE',
        });
    },
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CollectionsAPI,
        QueryAPI,
        ModelsAPI,
        SessionsAPI,
    };
}

