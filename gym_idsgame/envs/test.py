import os
import sys
import signal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from topology.topology import Mytopo
from containernet.net import Containernet
from mininet.node import Controller, OVSKernelSwitch, RemoteController
from containernet.cli import CLI
from containernet.link import TCLink
from mininet.log import info, setLogLevel
import subprocess

def stop_network(net):
    def handler(signal, frame):
        print("Stopping the network...")
        net.stop()
        exit(0)
    return handler
    

if __name__ == '__main__':
    setLogLevel('info')
    os.system('sudo sysctl -w net.ipv4.ip_forward=1')
    os.system('sudo iptables -t nat -A POSTROUTING -o enp0s3 -j MASQUERADE')
    os.system('ip route')
    os.system('sudo iptables -A FORWARD -i s1 -j ACCEPT')
    os.system('sudo iptables -A FORWARD -o s1 -j ACCEPT')
    os.system('sudo iptables -A FORWARD -i s3 -j ACCEPT')
    os.system('sudo iptables -A FORWARD -o s3 -j ACCEPT')
    
    topo = Mytopo()
    #print("Testing connectivity in my network")
    topo.net.pingAll()
    #CLI(topo.net)
    #topo.net.stop()
    signal.signal(signal.SIGINT, stop_network(topo.net))
    while True:
        pass
    os.system('sudo echo "nameserver 8.8.8.8" > /etc/resolv.conf')
