import xml.dom.minidom

def getXMLInfo(xmlFile):
    xmlDoc = xml.dom.minidom.parse(xmlFile)
    posX = xmlDoc.getElementsByTagName("MyCar")[0].getElementsByTagName("Position")[0].getElementsByTagName('X')[0].firstChild.nodeValue
    posY = xmlDoc.getElementsByTagName("MyCar")[0].getElementsByTagName("Position")[0].getElementsByTagName('Y')[0].firstChild.nodeValue
    theta = xmlDoc.getElementsByTagName("MyCar")[0].getElementsByTagName("Rotation")[0].getElementsByTagName('Z')[0].firstChild.nodeValue
    vel = xmlDoc.getElementsByTagName("MyCar")[0].getElementsByTagName("Speed")[0].firstChild.nodeValue
    steerAngle = xmlDoc.getElementsByTagName("MyCar")[0].getElementsByTagName("SteeringAngle")[0].firstChild.nodeValue
    dt = xmlDoc.getElementsByTagName("LastFrameTime")[0].firstChild.nodeValue
    info = [posX, posY, theta, vel, steerAngle, dt]
    for i in range(len(info)):
        info[i] = float(info[i])
    return info
