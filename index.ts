import cors from "cors";
import dotenv from "dotenv";
import express, {
  type NextFunction,
  type Request,
  type Response,
} from "express";
import { createRemoteJWKSet, jwtVerify } from "jose-cjs";
import { MongoClient, ServerApiVersion, type Db } from "mongodb";

dotenv.config({ quiet: true });

/* =========================================================
   Environment variables
========================================================= */

const port = Number(process.env.PORT) || 5000;

const clientUrl = process.env.CLIENT_URL || "http://localhost:3000";

const betterAuthUrl = (
  process.env.BETTER_AUTH_URL || "http://localhost:3000"
).replace(/\/+$/, "");

const mongoDbUri = process.env.MONGODB_URI;
const mongoDbName = process.env.MONGODB_DB_NAME;

if (!mongoDbUri) {
  throw new Error("MONGODB_URI is missing from the .env file");
}

if (!mongoDbName) {
  throw new Error("MONGODB_DB_NAME is missing from the .env file");
}

/* =========================================================
   Express application
========================================================= */

const app = express();

/* =========================================================
   MongoDB configuration
========================================================= */

const mongoClient = new MongoClient(mongoDbUri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: false,
    deprecationErrors: true,
  },
});

let database: Db | null = null;

/* =========================================================
   Better Auth JWKS configuration
========================================================= */

const jwksUrl = new URL(`${betterAuthUrl}/api/auth/jwks`);

const jwks = createRemoteJWKSet(jwksUrl);

/* =========================================================
   Authentication types
========================================================= */

type UserRole = "admin" | "doctor" | "patient";

interface AuthenticatedRequest extends Request {
  userId?: string;
  userName?: string;
  userEmail?: string;
  userRole?: UserRole;
}

/* =========================================================
   Global middlewares
========================================================= */

app.use(
  cors({
    origin: clientUrl,
    credentials: true,
  }),
);

app.use(
  express.json({
    limit: "1mb",
  }),
);

app.use(express.urlencoded({ extended: true }));

/* =========================================================
   JWT verification middleware
========================================================= */

const verifyToken = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): Promise<void> => {
  const authorizationHeader = req.headers.authorization;

  if (!authorizationHeader) {
    res.status(401).json({
      success: false,
      message: "Authorization token is required",
    });

    return;
  }

  const [authorizationType, token] = authorizationHeader.split(" ");

  if (authorizationType !== "Bearer" || !token) {
    res.status(401).json({
      success: false,
      message: "A valid Bearer token is required",
    });

    return;
  }

  try {
    const { payload } = await jwtVerify(token, jwks);

    const authenticatedUserId =
      typeof payload.sub === "string"
        ? payload.sub
        : typeof payload.id === "string"
          ? payload.id
          : undefined;

    if (!authenticatedUserId) {
      res.status(403).json({
        success: false,
        message: "Token does not contain a valid user ID",
      });

      return;
    }

    req.userId = authenticatedUserId;

    req.userName = typeof payload.name === "string" ? payload.name : undefined;

    req.userEmail =
      typeof payload.email === "string" ? payload.email : undefined;

    next();
  } catch (error) {
    console.error(
      "JWT verification error:",
      error instanceof Error ? error.message : error,
    );

    res.status(403).json({
      success: false,
      message: "Invalid or expired access token",
    });
  }
};

/* =========================================================
   Role verification middleware
========================================================= */

const verifyRole = (requiredRole: UserRole) => {
  return async (
    req: AuthenticatedRequest,
    res: Response,
    next: NextFunction,
  ): Promise<void> => {
    try {
      if (!req.userId) {
        res.status(401).json({
          success: false,
          message: "Authentication is required before role verification",
        });

        return;
      }

      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const usersCollection = database.collection("user");

      const userQueryConditions: Record<string, unknown>[] = [
        {
          id: req.userId,
        },
      ];

      if (req.userEmail) {
        userQueryConditions.push({
          email: req.userEmail,
        });
      }

      const currentUser = await usersCollection.findOne({
        $or: userQueryConditions,
      });

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const currentRole = currentUser.role;

      const validRoles: UserRole[] = ["admin", "doctor", "patient"];

      if (
        typeof currentRole !== "string" ||
        !validRoles.includes(currentRole as UserRole)
      ) {
        res.status(403).json({
          success: false,
          message: "User role is missing or invalid",
        });

        return;
      }

      if (currentRole !== requiredRole) {
        res.status(403).json({
          success: false,
          message: `${requiredRole} access is required`,
        });

        return;
      }

      req.userRole = currentRole as UserRole;

      next();
    } catch (error) {
      console.error(
        "Role verification error:",
        error instanceof Error ? error.message : error,
      );

      res.status(500).json({
        success: false,
        message: "Failed to verify current user role",
      });
    }
  };
};

/* =========================================================
   Admin, doctor and patient middlewares
========================================================= */

const verifyAdmin = verifyRole("admin");

const verifyDoctor = verifyRole("doctor");

const verifyPatient = verifyRole("patient");

/* =========================================================
   Public root route
========================================================= */

app.get("/", (_req: Request, res: Response) => {
  res.status(200).json({
    success: true,
    message: "SebaSathi AI server is running",
  });
});

/* =========================================================
   Public health route
========================================================= */

app.get("/api/v1/health", async (_req: Request, res: Response) => {
  try {
    if (!database) {
      res.status(503).json({
        success: false,
        message: "Database is not connected",
      });

      return;
    }

    await database.command({ ping: 1 });

    res.status(200).json({
      success: true,
      message: "SebaSathi AI API is healthy",
      database: "connected",
      databaseName: database.databaseName,
      timestamp: new Date().toISOString(),
    });
  } catch {
    res.status(503).json({
      success: false,
      message: "MongoDB connection is unavailable",
      database: "disconnected",
      timestamp: new Date().toISOString(),
    });
  }
});

/* =========================================================
   Protected authentication test route
========================================================= */

app.get(
  "/api/v1/auth/me",
  verifyToken,
  (req: AuthenticatedRequest, res: Response) => {
    res.status(200).json({
      success: true,
      message: "Authenticated user retrieved successfully",
      user: {
        id: req.userId,
        name: req.userName || null,
        email: req.userEmail || null,
      },
    });
  },
);

/*
  Admin API middleware:

  app.get(
    "/api/v1/admin/your-api",
    verifyToken,
    verifyAdmin,
    yourController
  );
*/

/*
  Doctor API middleware:

  app.get(
    "/api/v1/doctor/your-api",
    verifyToken,
    verifyDoctor,
    yourController
  );
*/

/*
  Patient API middleware:

  app.get(
    "/api/v1/patient/your-api",
    verifyToken,
    verifyPatient,
    yourController
  );
*/

/* =========================================================
   Unknown route handler
========================================================= */

app.use((_req: Request, res: Response) => {
  res.status(404).json({
    success: false,
    message: "API route not found",
  });
});

/* =========================================================
   Global error handler
========================================================= */

app.use((error: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error("Server error:", error);

  res.status(500).json({
    success: false,
    message: "Internal server error",
  });
});

/* =========================================================
   MongoDB connection
========================================================= */

const connectDatabase = async (): Promise<void> => {
  await mongoClient.connect();

  database = mongoClient.db(mongoDbName);

  await database.command({ ping: 1 });

  console.log(`MongoDB connected successfully. Database: ${mongoDbName}`);
};

/* =========================================================
   Start server
========================================================= */

const startServer = async (): Promise<void> => {
  try {
    await connectDatabase();

    app.listen(port, () => {
      console.log(`SebaSathi AI server is running on http://localhost:${port}`);

      console.log(`JWKS URL: ${jwksUrl.toString()}`);
    });
  } catch (error) {
    console.error(
      "Unable to start SebaSathi AI server:",
      error instanceof Error ? error.message : error,
    );

    await mongoClient.close();
    process.exit(1);
  }
};

void startServer();

/* =========================================================
   Graceful shutdown
========================================================= */

const shutdownServer = async (signal: string): Promise<void> => {
  console.log(`${signal} received. Closing MongoDB connection...`);

  try {
    await mongoClient.close();

    console.log("MongoDB connection closed successfully");

    process.exit(0);
  } catch (error) {
    console.error("Error closing MongoDB connection:", error);

    process.exit(1);
  }
};

process.on("SIGINT", () => {
  void shutdownServer("SIGINT");
});

process.on("SIGTERM", () => {
  void shutdownServer("SIGTERM");
});

export default app;
