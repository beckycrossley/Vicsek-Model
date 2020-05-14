#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vicsek_model_lib.py

Libs for the Vicsek Model.
"""
import numpy as np
from scipy import stats


def initialize(L,N,M=0):
    """Initialize the state of preys and predators.
    
    Parameters
    ----------
    L: float
        Size of the container(L*L)
    N : int
        Number of preys
    M : int, optional
        Number of predators

    Returns
    -------
    prey: ndarray
        State of prey(x,y,theta)
    predator: ndarray
        State of predator(x,y,theta)
    """
    # initial state of preys
    prey = np.zeros((N,3))
    prey[:,:2] = np.random.uniform(0,L,(N,2)) # positions x,y
    prey[:,2] = np.random.uniform(0,2*np.pi,N) # angles theta
    if M==0:   # there is no predator in the system
        return prey
    # initial state of predators
    predator = np.zeros((M,3))
    predator[:,:2] = np.random.uniform(0,L,(M,2))  # positions x,y
    predator[:,2] = np.random.uniform(0,2*np.pi,M)  # angles theta
    return prey, predator

def cal_pdist(pos,L):
    """Calculate the distance matrix(N*N) calculated from birds positions.
    
    Parameters
    ----------
    pos: ndarray
        the position matrix(N*2) of birds(x,y)
    L: float
        Size of the container(L*L)

    Returns
    -------
    Dist: ndarray
        the distance matrix(symmetric) calculated from position vectors of birds under periodic boundary conditions
    """
    N = np.shape(pos)[0]     # number of birds
    Dist = np.zeros((N,N))
    # calculate the distance between ith bird p1=(x1,y1) jth bird p2=(x2,y2) under periodic boundary conditions
    # use the technique of vectorization
    for i in range(N):
        dx = np.abs(pos[:,0]-pos[i,0])
        dy = np.abs(pos[:,1]-pos[i,1])
        dx[dx > L/2] -= L
        dy[dy > L/2] -= L
        Dist[i,:] = np.sqrt(dx**2 + dy**2)
    return Dist


def update_wop(state, L, v0, R, sigma): 
    """Update the state of birds in each time step without predator.
    
    Parameters
    ----------
    prey: ndarray
        State matrix of birds(x,y,theta)
    L: float
        Size of the container(L*L)
    v0: float
        The constant velocity of the birds per time step
    R: float
        Radius within which for preys to search for neighbours
    sigma: float
        Standard deviation of noise term for the orientation of preys

    Returns
    -------
    state: ndarray
        State matrix of birds(x,y,theta)
    order_para: float
        The Vicsek order parameter at this time step
    """
    N = np.shape(state)[0]   # number of birds
    
    # Update positions(mod L) of birds using angles of last time step
    state[:,0] = (state[:,0] + v0*np.cos(state[:,2]))%L
    state[:,1] = (state[:,1] + v0*np.sin(state[:,2]))%L
        
    # Use adjacency matrix to determine neighbours
    A = cal_pdist(state[:,:2],L)
    # Initialize heading
    heading = np.zeros(N)
    
    for i in range(N):
        ## check neighbouring brids within radius R ##
        adj = np.where(A[i,:] < R)[0] # indices of adjacent birds within radius R
        theta = state[adj,2] # angles of all adjacent birds
        
        # Sum sin and cos of angles
        sum_sin = np.sum(np.sin(theta))
        sum_cos = np.sum(np.cos(theta))
        
        # Compute heading for ith bird
        heading[i] = np.arctan2(sum_sin, sum_cos)
        
    # Update angle of birds with new headings and noise term
    state[:,2] = heading + stats.norm(0,sigma).rvs(N)

    # calculate the vicsek order parameter
    x = np.sum(np.cos(state[:,2]))
    y = np.sum(np.sin(state[:,2]))
    order_para = np.sqrt(x**2+y**2)/N
         
    return state, order_para
 
def update_pred(prey, predator, L, v0, v_predator, R, R_predator, R_run_away, sigma_prey, sigma_predator): 
    """Update the state of preys and predators in each time step with predators in the system.
    
    Parameters
    ----------
    prey: ndarray
        State matrix of preys(x,y,theta)
    predator: ndarray
        State matrix of predators(x,y,theta)
    L: float
        Size of the container(L*L)
    v0: float
        The constant velocity of the preys per time step
    v_predator: float
        The constant velocity of the preditors per time step
    R: float
        Radius within which for preys to search for neighbours
    R_predator: float
        Radius within which for predators to search for preys
    R_run_away: float
        Radius within which for preys to spot the predator and run away
    sigma_prey: float
        Standard deviation of noise term for the orientation of preys
    sigma_predator: float
        Standard deviation of noise term for the orientation of predators

    Returns
    -------
    prey: ndarray
        State matrix of preys(x,y,theta)
    predator: ndarray
        State matrix of predators(x,y,theta)
    order_para: float
        The Vicsek order paramter at this time step
    """
    N = np.shape(prey)[0]   # number of preys
    M = np.shape(predator)[0]  # number of predators
        
    # Use adjacency matrix to determine neighbours
    birds = np.concatenate((prey,predator),axis=0)
    A = cal_pdist(birds[:,:2],L)

    ## Remove the dead preys from the system ##
    I = np.argmin(A[N:,:N],axis=1)  # the indices of the preys nearest to the M predators respectively
    D = np.amin(A[N:,:N],axis=1)  # the distances between the nearest preys and the corresponding M predators respectively
    dead_prey_index = np.unique(I[D<(v_predator-v0)/2])
    
    prey = np.delete(prey, dead_prey_index, axis=0)  # delete the dead bird
    A = np.delete(A, dead_prey_index, axis=0)
    A = np.delete(A, dead_prey_index, axis=1)
    N = N - len(dead_prey_index)   # number of preys left decreases by 1     
    
    ## update the state of preys ##
    
    # Update positions(mod L) of preys using angles of last time step
    prey[:,0] = (prey[:,0] + v0*np.cos(prey[:,2]))%L
    prey[:,1] = (prey[:,1] + v0*np.sin(prey[:,2]))%L
    
    # Initialize heading of preys
    heading = np.zeros(N)
    
    for i in range(N):
        ## check neighbouring preys within radius R ##
        adj = np.where(A[i,:N] < R)[0] # indices of preys within radius R

        theta = prey[adj,2] # angles of all adjacent preys
        
        # Sum sin and cos of angles
        sum_sin = np.sum(np.sin(theta))
        sum_cos = np.sum(np.cos(theta))
        
        # Compute heading for ith prey
        heading[i] = np.arctan2(sum_sin, sum_cos)

        
        ## check neighbouring predators within radius R_run_away ##
        adj = np.where(A[i,N:] < R_run_away)[0]  # indices of predators within radius R_run_away
        if len(adj) > 0:
            dx = predator[adj,0] - prey[i,0]
            dy = predator[adj,1] - prey[i,1]
            dx[dx > L/2] -= L
            dx[dx < -L/2] += L
            dy[dy > L/2] -= L
            dy[dy < -L/2] += L  
            dist = np.sqrt(dx**2 + dy**2)
            run_away_angle = np.arctan2(-dy,-dx)
            sum_sin = np.sum(-np.log(dist/R_run_away)*np.sin(run_away_angle)) + np.sin(heading[i])
            sum_cos = np.sum(-np.log(dist/R_run_away)*np.cos(run_away_angle)) + np.cos(heading[i])
            heading[i] = np.arctan2(sum_sin, sum_cos)
     
    # Update angle of preys with new headings and noise term
    prey[:,2] = heading + stats.norm(0,sigma_prey).rvs(N)
    
    # calculate the Vicsek order parameter
    x = np.sum(np.cos(prey[:,2]))
    y = np.sum(np.sin(prey[:,2]))
    order_para = np.sqrt(x**2+y**2)/N
         
    ## update the state of predators ##    
    
    # Update positions(mod L) of predators using angles of last time step
    predator[:,0] = (predator[:,0] + v_predator*np.cos(predator[:,2]))%L
    predator[:,1] = (predator[:,1] + v_predator*np.sin(predator[:,2]))%L
    
    for j in range(M):
        ## check neighbouring preys within radius R_predator ##
        adj = np.where(A[N+j,:N] < R_predator)[0]  # indices of preys within radius R_predator
        if len(adj) > 0:  # if there is no prey within radius R_predator, the jth predator keeps in the same direction as the last time step
            dx = predator[j,0] - prey[adj,0]
            dy = predator[j,1] - prey[adj,1]
            dx[dx > L/2] -= L
            dx[dx < -L/2] += L
            dy[dy > L/2] -= L
            dy[dy < -L/2] += L  
            dist = np.sqrt(dx**2 + dy**2)
            run_after_angle = np.arctan2(-dy,-dx)
            sum_sin = np.sum(-np.log(dist/R_predator)*np.sin(run_after_angle))
            sum_cos = np.sum(-np.log(dist/R_predator)*np.cos(run_after_angle))
            predator[j,2] = np.arctan2(sum_sin, sum_cos)
      
    # Update angle of predators with noise term
    predator[:,2] += stats.norm(0,sigma_predator).rvs(M)
    
    return prey, predator, order_para