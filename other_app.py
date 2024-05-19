import os
import http.server
import socketserver

# Define the path to your React build directory
frontend_dir = os.path.join(os.getcwd(), 'build')

# Change directory to the frontend directory
os.chdir(frontend_dir)

# Set up a simple HTTP server
PORT = 3000
Handler = http.server.SimpleHTTPRequestHandler

# Serve the React build files
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Serving React build at http://localhost:{}".format(PORT))
    httpd.serve_forever()