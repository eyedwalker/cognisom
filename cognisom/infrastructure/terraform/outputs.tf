output "app_url" {
  description = "Application URL"
  value       = "https://${var.domain_name}"
}

output "alb_dns_name" {
  description = "ALB DNS name (use before domain is configured)"
  value       = aws_lb.main.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository URL for docker push"
  value       = aws_ecr_repository.main.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Active ECS service name"
  value       = var.enable_gpu ? aws_ecs_service.gpu[0].name : aws_ecs_service.cpu[0].name
}

output "mode" {
  description = "Current deployment mode"
  value       = var.enable_gpu ? "gpu (g4dn.xlarge spot)" : "cpu (Fargate)"
}

output "nameservers" {
  description = "Route53 nameservers - set these in GoDaddy"
  value       = aws_route53_zone.main.name_servers
}

output "s3_exports_bucket" {
  description = "S3 bucket for data exports"
  value       = aws_s3_bucket.exports.id
}

output "estimated_monthly_cost" {
  description = "Rough monthly cost estimate"
  value       = var.enable_gpu ? "~$170/month (GPU spot)" : "~$83/month (CPU Fargate)"
}

output "nameserver_instructions" {
  description = "Step-by-step GoDaddy DNS instructions"
  value       = <<-EOT

    ╔══════════════════════════════════════════════════════╗
    ║  Point cognisom.com to AWS (one-time setup)         ║
    ╠══════════════════════════════════════════════════════╣
    ║                                                      ║
    ║  1. Go to Squarespace Domains > cognisom.com > DNS    ║
    ║  2. Find DNS / Nameserver settings                   ║
    ║  3. Change nameservers to:                           ║
    ║     ${join("\n    ║     ", aws_route53_zone.main.name_servers)}
    ║                                                      ║
    ║  4. Save and wait 24-48 hours for propagation        ║
    ║  5. https://cognisom.com will then serve your app    ║
    ╚══════════════════════════════════════════════════════╝

  EOT
}
