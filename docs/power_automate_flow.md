# Power Automate Notification Flow

## Purpose

Notify the project owner by email whenever the cloud churn prediction file is modified in OneDrive.

## Flow

```text
OneDrive file modified
        ↓
Power Automate trigger
        ↓
Gmail sends notification email
```

## Configuration

- Flow name: `Notify when churn predictions update - Gmail`
- Trigger: OneDrive for Business — When a file is modified (properties only)
- Folder: `/customer-churn-pipeline`
- Action: Gmail — Send email (V2)
- Recipient: `nahom.mersha.alehgne@gmail.com`
- Subject: `Customer churn predictions updated`

## Verification

The flow was tested successfully by modifying the OneDrive prediction file.

The notification email was received successfully with the message:

> The cloud churn prediction file was modified in OneDrive.

## Important limitation

The local VS Code file is not automatically synchronized with OneDrive. The generated CSV must currently be uploaded or replaced in OneDrive manually before the notification is triggered.

The original Microsoft Mail connector was unavailable because it is restricted for new Microsoft tenants. Gmail was used successfully as an alternative connector.

## Completed pipeline

```text
Snowflake
    ↓
Python model inference
    ↓
cloud_predictions.csv
    ↓
OneDrive
    ↓
Power Automate
    ↓
Gmail notification
```
