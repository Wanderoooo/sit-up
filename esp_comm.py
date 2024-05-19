import socket
import keyboard

ip = "192.168.137.89"
def send_udp_message(message, ip, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.sendto(message.encode(), (ip, port))
    client_socket.close()

def on_spacebar_press(e):
    send_udp_message("buzz", ip, 12345)

def send_buzz():
    send_udp_message("buzz", ip, 12345)

if __name__ == "__main__":
    
    keyboard.on_press_key("space", on_spacebar_press)
    keyboard.wait()