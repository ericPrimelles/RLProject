from pysc2.lib import actions, features, units
import numpy as np
import math

def state_of_marine(marine, beacon, screen, distance_window):
    dist_x = beacon.x - marine.x
    dist_y = beacon.y - marine.y
    return discretize_distance(dist_x, screen, distance_window), discretize_distance(dist_y, screen, distance_window, 0.8)

def discretize_distance(dist, screen, distance_window, factor=1.0):
    if distance_window == -1:
        return discretize_distance_float(dist, screen, factor)

    percentual_val = round(dist / screen, 2)
    disc_val = math.ceil(percentual_val / distance_window * 100)
    return disc_val

def discretize_distance_float(dist, screen, factor=1.0):
    disc_dist = dist/(screen * factor)
    return disc_dist

def move_to_position(action, screen_size):
    # Definición de la posibles acciones
    dest = []
    if action == 0:
        dest = [0, screen_size] # arriba
    elif action == 1:
        dest = [screen_size, 0] # derecha
    elif action == 2:
        dest = [0, -screen_size] # abajo
    elif action == 3:
        dest = [-screen_size, 0] # izquierda
    elif action == 4:
        dest = [screen_size, screen_size] # derecha arriba 
    elif action == 5:
        dest = [screen_size, -screen_size] # derecha abajo 
    elif action == 6:
        dest = [-screen_size, screen_size] # izquierda arriba 
    elif action == 7:
        dest = [-screen_size, -screen_size] # izquierda abajo

    return dest