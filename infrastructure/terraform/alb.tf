# ─── Application Load Balancer ────────────────────────────────
resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = { Name = "${local.name_prefix}-alb" }
}

# ─── Target Groups ───────────────────────────────────────────
resource "aws_lb_target_group" "streamlit" {
  name        = "${var.project_name}-streamlit"
  port        = var.streamlit_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = var.enable_gpu ? "instance" : "ip"

  health_check {
    path                = "/api/health"
    port                = tostring(var.flask_port)
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
  }

  stickiness {
    type            = "lb_cookie"
    enabled         = true
    cookie_duration = 86400
  }

  tags = { Name = "${var.project_name}-streamlit-tg" }
}

resource "aws_lb_target_group" "flask" {
  name        = "${var.project_name}-flask"
  port        = var.flask_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = var.enable_gpu ? "instance" : "ip"

  health_check {
    path                = "/api/health"
    port                = tostring(var.flask_port)
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
  }

  tags = { Name = "${var.project_name}-flask-tg" }
}

# ─── HTTPS Listener (default → Streamlit) ────────────────────
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate_validation.main.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.streamlit.arn
  }
}

# ─── /api/* → Flask ──────────────────────────────────────────
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.flask.arn
  }

  condition {
    path_pattern { values = ["/api/*"] }
  }
}

# ─── /_stcore/* → Streamlit WebSocket ────────────────────────
resource "aws_lb_listener_rule" "streamlit_ws" {
  listener_arn = aws_lb_listener.https.arn
  priority     = 50

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.streamlit.arn
  }

  condition {
    path_pattern { values = ["/_stcore/*"] }
  }
}

# ─── HTTP → HTTPS Redirect ──────────────────────────────────
resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}
