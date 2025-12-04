# Frontend Demo

This is a single-file React component (App.jsx) for the demo. To run locally:
1. Create a React app (`npx create-react-app my-app`) and replace `src/App.js` with the contents of `App.jsx`.
2. Run the app (`npm start`) and use a proxy in package.json to forward `/api` to the backend (http://localhost:8000).

Proxy example in package.json:
"proxy": "http://localhost:8000"
