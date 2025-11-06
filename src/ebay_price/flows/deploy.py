from __future__ import annotations

from prefect.client.schemas.schedules import CronSchedule
from prefect.deployments import Deployment

# Import your full pipeline flow
# Adjust import if your flow is declared elsewhere
from ebay_price.flows.train_flow import main_flow  # noqa: F401


def build_deployment() -> Deployment:
    """
    Build a Prefect deployment running the full pipeline on a cron schedule.
    Default: daily at 02:00 local time (adjust as needed).
    """
    return Deployment.build_from_flow(
        flow=main_flow,
        name="daily-train",
        schedule=(CronSchedule(cron="0 2 * * *", timezone="localtime")),
        work_queue_name="default",
        parameters={},  # add flow params if you expose any
        tags=["price-prediction", "etl-train-log-explain"],
    )


if __name__ == "__main__":
    dep = build_deployment()
    dep.apply()
    print("Deployment 'daily-train' applied. Start an agent to run it.")
