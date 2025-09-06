variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "t3ss"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge", "m5.2xlarge"]
}

variable "gpu_node_instance_types" {
  description = "EC2 instance types for GPU nodes"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge", "p3.16xlarge"]
}

variable "node_desired_size" {
  description = "Desired number of EKS nodes"
  type        = number
  default     = 3
}

variable "node_max_size" {
  description = "Maximum number of EKS nodes"
  type        = number
  default     = 10
}

variable "node_min_size" {
  description = "Minimum number of EKS nodes"
  type        = number
  default     = 1
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 3
}

variable "redis_auth_token" {
  description = "Redis auth token"
  type        = string
  sensitive   = true
}

variable "opensearch_instance_type" {
  description = "OpenSearch instance type"
  type        = string
  default     = "r6g.large.search"
}

variable "opensearch_instance_count" {
  description = "Number of OpenSearch instances"
  type        = number
  default     = 3
}

variable "opensearch_master_instance_type" {
  description = "OpenSearch master instance type"
  type        = string
  default     = "r6g.medium.search"
}

variable "opensearch_volume_size" {
  description = "OpenSearch volume size in GB"
  type        = number
  default     = 100
}

variable "opensearch_password" {
  description = "OpenSearch admin password"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "t3ss.qisslabs.com"
}

variable "certificate_arn" {
  description = "ACM certificate ARN for SSL"
  type        = string
  default     = ""
}

variable "enable_monitoring" {
  description = "Enable monitoring stack"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable logging stack"
  type        = bool
  default     = true
}

variable "enable_backup" {
  description = "Enable backup configuration"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention in days"
  type        = number
  default     = 30
}

variable "enable_autoscaling" {
  description = "Enable cluster autoscaling"
  type        = bool
  default     = true
}

variable "enable_gpu_support" {
  description = "Enable GPU support"
  type        = bool
  default     = true
}

variable "enable_istio" {
  description = "Enable Istio service mesh"
  type        = bool
  default     = true
}

variable "enable_argo_cd" {
  description = "Enable ArgoCD for GitOps"
  type        = bool
  default     = true
}

variable "enable_external_secrets" {
  description = "Enable External Secrets Operator"
  type        = bool
  default     = true
}

variable "enable_cert_manager" {
  description = "Enable cert-manager for SSL certificates"
  type        = bool
  default     = true
}

variable "enable_nginx_ingress" {
  description = "Enable NGINX Ingress Controller"
  type        = bool
  default     = true
}

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "enable_jaeger" {
  description = "Enable Jaeger tracing"
  type        = bool
  default     = true
}

variable "enable_elasticsearch" {
  description = "Enable Elasticsearch for logging"
  type        = bool
  default     = true
}

variable "enable_kibana" {
  description = "Enable Kibana for log visualization"
  type        = bool
  default     = true
}

variable "enable_fluentd" {
  description = "Enable Fluentd for log collection"
  type        = bool
  default     = true
}

variable "enable_fluent_bit" {
  description = "Enable Fluent Bit for log collection"
  type        = bool
  default     = true
}

variable "enable_aws_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "enable_cluster_autoscaler" {
  description = "Enable Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "enable_aws_ebs_csi_driver" {
  description = "Enable AWS EBS CSI Driver"
  type        = bool
  default     = true
}

variable "enable_aws_efs_csi_driver" {
  description = "Enable AWS EFS CSI Driver"
  type        = bool
  default     = true
}

variable "enable_aws_fsx_csi_driver" {
  description = "Enable AWS FSx CSI Driver"
  type        = bool
  default     = false
}

variable "enable_aws_cloudwatch_agent" {
  description = "Enable AWS CloudWatch Agent"
  type        = bool
  default     = true
}

variable "enable_aws_xray_daemon" {
  description = "Enable AWS X-Ray Daemon"
  type        = bool
  default     = true
}

variable "enable_aws_node_termination_handler" {
  description = "Enable AWS Node Termination Handler"
  type        = bool
  default     = true
}

variable "enable_aws_pod_identity_webhook" {
  description = "Enable AWS Pod Identity Webhook"
  type        = bool
  default     = true
}

variable "enable_aws_secrets_manager_csi_driver" {
  description = "Enable AWS Secrets Manager CSI Driver"
  type        = bool
  default     = true
}

variable "enable_aws_parameter_store_csi_driver" {
  description = "Enable AWS Parameter Store CSI Driver"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_logs" {
  description = "Enable AWS CloudWatch Logs"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_metrics" {
  description = "Enable AWS CloudWatch Metrics"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_alarms" {
  description = "Enable AWS CloudWatch Alarms"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_dashboards" {
  description = "Enable AWS CloudWatch Dashboards"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_insights" {
  description = "Enable AWS CloudWatch Insights"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_synthetics" {
  description = "Enable AWS CloudWatch Synthetics"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_rum" {
  description = "Enable AWS CloudWatch RUM"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_evidently" {
  description = "Enable AWS CloudWatch Evidently"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_application_insights" {
  description = "Enable AWS CloudWatch Application Insights"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_anomaly_detection" {
  description = "Enable AWS CloudWatch Anomaly Detection"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_composite_alarms" {
  description = "Enable AWS CloudWatch Composite Alarms"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_metric_streams" {
  description = "Enable AWS CloudWatch Metric Streams"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_metric_filters" {
  description = "Enable AWS CloudWatch Metric Filters"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_groups" {
  description = "Enable AWS CloudWatch Log Groups"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_streams" {
  description = "Enable AWS CloudWatch Log Streams"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_events" {
  description = "Enable AWS CloudWatch Log Events"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_insights" {
  description = "Enable AWS CloudWatch Log Insights"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_anomaly_detection" {
  description = "Enable AWS CloudWatch Log Anomaly Detection"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_composite_alarms" {
  description = "Enable AWS CloudWatch Log Composite Alarms"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_metric_streams" {
  description = "Enable AWS CloudWatch Log Metric Streams"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_metric_filters" {
  description = "Enable AWS CloudWatch Log Metric Filters"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_groups" {
  description = "Enable AWS CloudWatch Log Groups"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_streams" {
  description = "Enable AWS CloudWatch Log Streams"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_events" {
  description = "Enable AWS CloudWatch Log Events"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_insights" {
  description = "Enable AWS CloudWatch Log Insights"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_anomaly_detection" {
  description = "Enable AWS CloudWatch Log Anomaly Detection"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_composite_alarms" {
  description = "Enable AWS CloudWatch Log Composite Alarms"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_metric_streams" {
  description = "Enable AWS CloudWatch Log Metric Streams"
  type        = bool
  default     = true
}

variable "enable_aws_cloudwatch_log_metric_filters" {
  description = "Enable AWS CloudWatch Log Metric Filters"
  type        = bool
  default     = true
}