# Power Automate Notification Flow

## Purpose

Notify the project owner when the generated churn-prediction CSV is updated in OneDrive.

## Flow

```text
cloud_predictions.csv modified in OneDrive
        ↓
Power Automate trigger
        ↓
Gmail sends a generic update email
```

## Configuration

- Flow: `Notify when churn predictions update - Gmail`
- Trigger: OneDrive for Business — file modified (properties only)
- Folder: `/customer-churn-pipeline`
- Action: Gmail — Send email (V2)
- Subject: `Customer churn predictions updated`

## Verification

The flow was tested by modifying the OneDrive prediction file. The notification was received successfully.

Message body:

> The cloud churn prediction file was modified in OneDrive.

## Important boundary

The local VS Code file is not automatically connected to OneDrive. After the cloud pipeline creates a new CSV, the file must currently be uploaded or replaced in the OneDrive folder before this notification can run.

The Microsoft Mail connector was unavailable for the account, so Gmail was used as the working alternative.

## Privacy

The flow uses the public learning dataset and sends only a generic update message. It does not email customer records, Snowflake credentials, or model secrets.
