// T3SS Project
// File: shared_libs/service_discovery/service_registry.go
// (c) 2025 Qiss Labs. All Rights Reserved.
// Unauthorized copying or distribution of this file is strictly prohibited.
// For internal use only.

package service_discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"go.etcd.io/etcd/clientv3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ServiceRegistry manages service discovery and load balancing
type ServiceRegistry struct {
	etcdClient *clientv3.Client
	services   map[string][]*ServiceInstance
	mu         sync.RWMutex
	watchers   map[string]chan struct{}
	stopCh     chan struct{}
}

// ServiceInstance represents a service instance
type ServiceInstance struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Address     string            `json:"address"`
	Port        int               `json:"port"`
	Version     string            `json:"version"`
	Health      string            `json:"health"`
	LastSeen    time.Time         `json:"last_seen"`
	Metadata    map[string]string `json:"metadata"`
	Weight      int               `json:"weight"`
	Tags        []string          `json:"tags"`
	grpcConn    *grpc.ClientConn
}

// ServiceConfig represents service configuration
type ServiceConfig struct {
	Name        string            `json:"name"`
	Address     string            `json:"address"`
	Port        int               `json:"port"`
	Version     string            `json:"version"`
	Health      string            `json:"health"`
	Metadata    map[string]string `json:"metadata"`
	Weight      int               `json:"weight"`
	Tags        []string          `json:"tags"`
	TTL         time.Duration     `json:"ttl"`
}

// LoadBalancer interface for load balancing strategies
type LoadBalancer interface {
	SelectInstance(instances []*ServiceInstance) *ServiceInstance
}

// RoundRobinLoadBalancer implements round-robin load balancing
type RoundRobinLoadBalancer struct {
	current int
	mu      sync.Mutex
}

// WeightedRoundRobinLoadBalancer implements weighted round-robin load balancing
type WeightedRoundRobinLoadBalancer struct {
	current int
	weights []int
	mu      sync.Mutex
}

// LeastConnectionsLoadBalancer implements least connections load balancing
type LeastConnectionsLoadBalancer struct {
	connections map[string]int
	mu          sync.Mutex
}

// NewServiceRegistry creates a new service registry
func NewServiceRegistry(etcdEndpoints []string) (*ServiceRegistry, error) {
	client, err := clientv3.New(clientv3.Config{
		Endpoints:   etcdEndpoints,
		DialTimeout: 5 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %w", err)
	}

	registry := &ServiceRegistry{
		etcdClient: client,
		services:   make(map[string][]*ServiceInstance),
		watchers:   make(map[string]chan struct{}),
		stopCh:     make(chan struct{}),
	}

	// Start background tasks
	go registry.cleanupExpiredServices()
	go registry.watchServices()

	return registry, nil
}

// RegisterService registers a service instance
func (sr *ServiceRegistry) RegisterService(config *ServiceConfig) error {
	instance := &ServiceInstance{
		ID:       generateServiceID(config.Name, config.Address, config.Port),
		Name:     config.Name,
		Address:  config.Address,
		Port:     config.Port,
		Version:  config.Version,
		Health:   config.Health,
		LastSeen: time.Now(),
		Metadata: config.Metadata,
		Weight:   config.Weight,
		Tags:     config.Tags,
	}

	// Create gRPC connection
	conn, err := grpc.Dial(
		fmt.Sprintf("%s:%d", config.Address, config.Port),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return fmt.Errorf("failed to create gRPC connection: %w", err)
	}
	instance.grpcConn = conn

	// Serialize instance data
	data, err := json.Marshal(instance)
	if err != nil {
		return fmt.Errorf("failed to marshal service instance: %w", err)
	}

	// Register with etcd
	key := fmt.Sprintf("/services/%s/%s", config.Name, instance.ID)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Use TTL for automatic cleanup
	lease, err := sr.etcdClient.Grant(ctx, int64(config.TTL.Seconds()))
	if err != nil {
		return fmt.Errorf("failed to create lease: %w", err)
	}

	_, err = sr.etcdClient.Put(ctx, key, string(data), clientv3.WithLease(lease.ID))
	if err != nil {
		return fmt.Errorf("failed to register service: %w", err)
	}

	// Keep lease alive
	go sr.keepAlive(lease.ID)

	// Update local registry
	sr.mu.Lock()
	sr.services[config.Name] = append(sr.services[config.Name], instance)
	sr.mu.Unlock()

	log.Printf("Service %s registered at %s:%d", config.Name, config.Address, config.Port)
	return nil
}

// DeregisterService deregisters a service instance
func (sr *ServiceRegistry) DeregisterService(serviceName, instanceID string) error {
	key := fmt.Sprintf("/services/%s/%s", serviceName, instanceID)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := sr.etcdClient.Delete(ctx, key)
	if err != nil {
		return fmt.Errorf("failed to deregister service: %w", err)
	}

	// Remove from local registry
	sr.mu.Lock()
	if instances, exists := sr.services[serviceName]; exists {
		for i, instance := range instances {
			if instance.ID == instanceID {
				// Close gRPC connection
				if instance.grpcConn != nil {
					instance.grpcConn.Close()
				}
				// Remove from slice
				sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
				break
			}
		}
	}
	sr.mu.Unlock()

	log.Printf("Service %s instance %s deregistered", serviceName, instanceID)
	return nil
}

// GetServiceInstances returns all instances of a service
func (sr *ServiceRegistry) GetServiceInstances(serviceName string) ([]*ServiceInstance, error) {
	sr.mu.RLock()
	instances, exists := sr.services[serviceName]
	sr.mu.RUnlock()

	if !exists || len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for service %s", serviceName)
	}

	// Filter healthy instances
	var healthyInstances []*ServiceInstance
	for _, instance := range instances {
		if instance.Health == "healthy" {
			healthyInstances = append(healthyInstances, instance)
		}
	}

	if len(healthyInstances) == 0 {
		return nil, fmt.Errorf("no healthy instances found for service %s", serviceName)
	}

	return healthyInstances, nil
}

// GetServiceInstance returns a single instance using load balancing
func (sr *ServiceRegistry) GetServiceInstance(serviceName string, strategy LoadBalancer) (*ServiceInstance, error) {
	instances, err := sr.GetServiceInstances(serviceName)
	if err != nil {
		return nil, err
	}

	if strategy == nil {
		strategy = &RoundRobinLoadBalancer{}
	}

	return strategy.SelectInstance(instances), nil
}

// GetGRPCConnection returns a gRPC connection to a service instance
func (sr *ServiceRegistry) GetGRPCConnection(serviceName string, strategy LoadBalancer) (*grpc.ClientConn, error) {
	instance, err := sr.GetServiceInstance(serviceName, strategy)
	if err != nil {
		return nil, err
	}

	if instance.grpcConn == nil {
		// Create new connection if not exists
		conn, err := grpc.Dial(
			fmt.Sprintf("%s:%d", instance.Address, instance.Port),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create gRPC connection: %w", err)
		}
		instance.grpcConn = conn
	}

	return instance.grpcConn, nil
}

// WatchService watches for changes in a service
func (sr *ServiceRegistry) WatchService(serviceName string) <-chan []*ServiceInstance {
	ch := make(chan []*ServiceInstance, 1)
	
	sr.mu.Lock()
	sr.watchers[serviceName] = ch
	sr.mu.Unlock()

	// Send current instances
	instances, _ := sr.GetServiceInstances(serviceName)
	ch <- instances

	return ch
}

// Stop stops the service registry
func (sr *ServiceRegistry) Stop() {
	close(sr.stopCh)
	sr.etcdClient.Close()
}

// Load balancer implementations

// SelectInstance selects an instance using round-robin
func (rr *RoundRobinLoadBalancer) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
	if len(instances) == 0 {
		return nil
	}

	rr.mu.Lock()
	defer rr.mu.Unlock()

	instance := instances[rr.current]
	rr.current = (rr.current + 1) % len(instances)
	return instance
}

// SelectInstance selects an instance using weighted round-robin
func (wrr *WeightedRoundRobinLoadBalancer) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
	if len(instances) == 0 {
		return nil
	}

	wrr.mu.Lock()
	defer wrr.mu.Unlock()

	// Initialize weights if not set
	if len(wrr.weights) != len(instances) {
		wrr.weights = make([]int, len(instances))
		for i, instance := range instances {
			wrr.weights[i] = instance.Weight
		}
	}

	// Find instance with highest current weight
	maxWeight := -1
	selectedIndex := 0

	for i, instance := range instances {
		wrr.weights[i] += instance.Weight
		if wrr.weights[i] > maxWeight {
			maxWeight = wrr.weights[i]
			selectedIndex = i
		}
	}

	// Decrease weight of selected instance
	wrr.weights[selectedIndex] -= maxWeight

	return instances[selectedIndex]
}

// SelectInstance selects an instance using least connections
func (lc *LeastConnectionsLoadBalancer) SelectInstance(instances []*ServiceInstance) *ServiceInstance {
	if len(instances) == 0 {
		return nil
	}

	lc.mu.Lock()
	defer lc.mu.Unlock()

	// Initialize connections map if not set
	if lc.connections == nil {
		lc.connections = make(map[string]int)
	}

	// Find instance with least connections
	minConnections := int(^uint(0) >> 1) // Max int
	selectedInstance := instances[0]

	for _, instance := range instances {
		connections := lc.connections[instance.ID]
		if connections < minConnections {
			minConnections = connections
			selectedInstance = instance
		}
	}

	// Increment connection count
	lc.connections[selectedInstance.ID]++

	return selectedInstance
}

// DecrementConnections decrements connection count for an instance
func (lc *LeastConnectionsLoadBalancer) DecrementConnections(instanceID string) {
	lc.mu.Lock()
	defer lc.mu.Unlock()

	if lc.connections == nil {
		return
	}

	if count := lc.connections[instanceID]; count > 0 {
		lc.connections[instanceID]--
	}
}

// Helper methods

// keepAlive keeps the lease alive
func (sr *ServiceRegistry) keepAlive(leaseID clientv3.LeaseID) {
	ch, kaerr := sr.etcdClient.KeepAlive(context.Background(), leaseID)
	if kaerr != nil {
		log.Printf("Failed to keep lease alive: %v", kaerr)
		return
	}

	for {
		select {
		case <-sr.stopCh:
			return
		case _, ok := <-ch:
			if !ok {
				log.Printf("Keep alive channel closed for lease %d", leaseID)
				return
			}
		}
	}
}

// cleanupExpiredServices removes expired services from local registry
func (sr *ServiceRegistry) cleanupExpiredServices() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-sr.stopCh:
			return
		case <-ticker.C:
			sr.mu.Lock()
			for serviceName, instances := range sr.services {
				var activeInstances []*ServiceInstance
				for _, instance := range instances {
					if time.Since(instance.LastSeen) < 2*time.Minute {
						activeInstances = append(activeInstances, instance)
					} else {
						// Close expired connection
						if instance.grpcConn != nil {
							instance.grpcConn.Close()
						}
					}
				}
				sr.services[serviceName] = activeInstances
			}
			sr.mu.Unlock()
		}
	}
}

// watchServices watches for service changes in etcd
func (sr *ServiceRegistry) watchServices() {
	watchCh := sr.etcdClient.Watch(context.Background(), "/services/", clientv3.WithPrefix())

	for {
		select {
		case <-sr.stopCh:
			return
		case watchResp := <-watchCh:
			for _, event := range watchResp.Events {
				sr.handleServiceChange(event)
			}
		}
	}
}

// handleServiceChange handles service changes from etcd
func (sr *ServiceRegistry) handleServiceChange(event *clientv3.Event) {
	// Parse service name from key
	key := string(event.Kv.Key)
	parts := strings.Split(key, "/")
	if len(parts) < 3 {
		return
	}

	serviceName := parts[2]

	switch event.Type {
	case clientv3.EventTypePut:
		// Service added or updated
		var instance ServiceInstance
		if err := json.Unmarshal(event.Kv.Value, &instance); err != nil {
			log.Printf("Failed to unmarshal service instance: %v", err)
			return
		}

		sr.mu.Lock()
		// Update or add instance
		found := false
		for i, existing := range sr.services[serviceName] {
			if existing.ID == instance.ID {
				sr.services[serviceName][i] = &instance
				found = true
				break
			}
		}
		if !found {
			sr.services[serviceName] = append(sr.services[serviceName], &instance)
		}
		sr.mu.Unlock()

		// Notify watchers
		sr.notifyWatchers(serviceName)

	case clientv3.EventTypeDelete:
		// Service removed
		instanceID := parts[3]

		sr.mu.Lock()
		if instances, exists := sr.services[serviceName]; exists {
			for i, instance := range instances {
				if instance.ID == instanceID {
					// Close connection
					if instance.grpcConn != nil {
						instance.grpcConn.Close()
					}
					// Remove from slice
					sr.services[serviceName] = append(instances[:i], instances[i+1:]...)
					break
				}
			}
		}
		sr.mu.Unlock()

		// Notify watchers
		sr.notifyWatchers(serviceName)
	}
}

// notifyWatchers notifies watchers of service changes
func (sr *ServiceRegistry) notifyWatchers(serviceName string) {
	sr.mu.RLock()
	watcher, exists := sr.watchers[serviceName]
	sr.mu.RUnlock()

	if exists {
		instances, _ := sr.GetServiceInstances(serviceName)
		select {
		case watcher <- instances:
		default:
			// Channel is full, skip notification
		}
	}
}

// generateServiceID generates a unique service ID
func generateServiceID(name, address string, port int) string {
	return fmt.Sprintf("%s-%s-%d-%d", name, address, port, time.Now().Unix())
}