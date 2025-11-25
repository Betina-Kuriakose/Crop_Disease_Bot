# MongoDB Setup Guide

## Automatic Database & Collection Creation

**You don't need to manually create anything!** The Flask app will automatically create the database and collection when it starts.

## What Gets Created

When you run `python app_flask.py`, the app will:

1. **Connect to MongoDB** using the URI from your `.env` file
2. **Create Database**: `crop_advisor` (or whatever you set in `MONGO_DB_NAME`)
3. **Create Collection**: `users` (or whatever you set in `MONGO_USER_COLLECTION`)
4. **Create Index**: Unique index on `username` field

## Default Configuration

- **Database Name**: `crop_advisor`
- **Collection Name**: `users`

You can change these by setting environment variables:
- `MONGO_DB_NAME` - Database name (default: `crop_advisor`)
- `MONGO_USER_COLLECTION` - Collection name (default: `users`)

## Collection Structure

The `users` collection will store documents like this:

```javascript
{
  "_id": ObjectId("..."),
  "username": "farmer_admin",
  "password_hash": "$2b$12$...",  // Hashed password
  "role": "admin" | "user",
  "created_at": "2024-01-01T12:00:00.000000"
}
```

## Verification

When the app starts, you'll see messages like:

```
[MongoDB] Attempting connection to: mongodb+srv://user:***@cluster0...
✅ Created/verified index on 'users.username'
✅ Collection 'users' exists
✅ MongoDB database connection established successfully
   Database: crop_advisor
   Collection: users
   Ready to store user credentials!
```

## Manual Verification (Optional)

If you want to check in MongoDB Atlas or Compass:

1. **Database**: Look for `crop_advisor`
2. **Collection**: Look for `users`
3. **Index**: Check that `username` has a unique index

## First User

- The first user you create via signup will be stored in the `users` collection
- If the collection is completely empty, a default admin user may be created:
  - Username: `farmer_admin` (or `DEFAULT_ADMIN_USER` env var)
  - Password: `demo123` (or `DEFAULT_ADMIN_PASS` env var)

## Troubleshooting

### Collection doesn't appear in MongoDB Atlas
- This is normal! MongoDB Atlas may not show collections until they contain at least one document
- The collection will appear after the first user signs up

### Connection errors
- Check your `.env` file has the correct MongoDB URI
- Ensure your MongoDB Atlas cluster allows connections from your IP
- Verify your MongoDB username and password are correct

