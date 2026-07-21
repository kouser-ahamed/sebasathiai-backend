import cors from "cors";
import dotenv from "dotenv";
import express, {
  type NextFunction,
  type Request,
  type Response,
} from "express";
import { createRemoteJWKSet, jwtVerify } from "jose-cjs";
import {
  MongoClient,
  ObjectId,
  ServerApiVersion,
  type Db,
  type Document,
  type Filter,
} from "mongodb";

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

const groqApiKey = process.env.GROQ_API_KEY;

const groqApiBaseUrl = (
  process.env.GROQ_API_BASE_URL || "https://api.groq.com/openai/v1"
).replace(/\/+$/, "");

const groqModel = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

const AI_HEALTH_HISTORY_COLLECTION = "AI-health-History";

const AI_HEALTH_CHAT_COLLECTION = "all-history";

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
   NOTE: serverSelectionTimeoutMS is set so that a bad/unreachable
   Mongo URI fails fast (a few seconds) instead of hanging the
   serverless function until Vercel kills it.
========================================================= */

const mongoClient = new MongoClient(mongoDbUri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: false,
    deprecationErrors: true,
  },
  maxPoolSize: 10,
  serverSelectionTimeoutMS: 8000,
});

let database: Db | null = null;

/* =========================================================
   Better Auth JWKS configuration
========================================================= */

const jwksUrl = new URL(`${betterAuthUrl}/api/auth/jwks`);

const jwks = createRemoteJWKSet(jwksUrl);

/* =========================================================
   Lazy, non-crashing MongoDB connection

   IMPORTANT (Vercel fix):
   The previous version awaited connectDatabase() once at module
   load and called process.exit(1) if it failed. On a serverless
   platform this KILLS the whole function process on any transient
   Mongo error, which is exactly what produced the
   "Node.js process exited with exit status: 1" + cascading 503
   errors seen in the logs.

   Fix: connect lazily, cache the connection promise so we only
   connect once per warm container, and NEVER exit the process.
   A failed connection simply returns a clean 503 response and the
   next request tries again.
========================================================= */

const createDatabaseIndexes = async (db: Db): Promise<void> => {
  await Promise.all([
    db.collection("user").createIndex({ role: 1, status: 1, updatedAt: -1 }),
    db.collection("user").createIndex({ role: 1, name: 1 }),
    db.collection("user").createIndex({ role: 1, email: 1 }),
    db
      .collection("doctors")
      .createIndex({ status: 1, ratingAverage: -1, createdAt: -1 }),
    db.collection("doctors").createIndex({ name: 1 }),
    db.collection("doctors").createIndex({ specialization: 1 }),
    db.collection("doctors").createIndex({ qualification: 1 }),
    db.collection("doctors").createIndex({ hospital: 1 }),
    db.collection("doctors").createIndex({ experienceYears: 1 }),
    db
      .collection("reviews")
      .createIndex({ doctorId: 1, userId: 1 }, { unique: true }),
    db.collection("reviews").createIndex({ doctorId: 1, updatedAt: -1 }),
    db
      .collection("appointments")
      .createIndex({ patientUserId: 1, doctorId: 1, status: 1 }),
    db
      .collection("appointments")
      .createIndex({ patientUserId: 1, createdAt: -1 }),
    db
      .collection("appointments")
      .createIndex({ doctorUserId: 1, status: 1, appointmentDate: 1 }),
    db
      .collection("appointments")
      .createIndex({ doctorUserId: 1, createdAt: -1 }),
    db
      .collection("appointments")
      .createIndex({ status: 1, appointmentDate: 1, appointmentTime: 1 }),
    db
      .collection(AI_HEALTH_HISTORY_COLLECTION)
      .createIndex({ userId: 1, createdAt: -1 }),
    db
      .collection(AI_HEALTH_HISTORY_COLLECTION)
      .createIndex({ patientUserId: 1, createdAt: -1 }),
    db
      .collection(AI_HEALTH_HISTORY_COLLECTION)
      .createIndex({ userId: 1, updatedAt: -1 }),
    db
      .collection(AI_HEALTH_HISTORY_COLLECTION)
      .createIndex({ patientUserId: 1, updatedAt: -1 }),
    db
      .collection(AI_HEALTH_CHAT_COLLECTION)
      .createIndex({ userId: 1, lastMessageAt: -1 }),
    db
      .collection(AI_HEALTH_CHAT_COLLECTION)
      .createIndex({ patientUserId: 1, lastMessageAt: -1 }),
  ]);
};

const connectDatabase = async (): Promise<Db> => {
  await mongoClient.connect();

  const db = mongoClient.db(mongoDbName);

  await db.command({ ping: 1 });

  await createDatabaseIndexes(db);

  console.log(`MongoDB connected successfully. Database: ${mongoDbName}`);

  return db;
};

let databaseConnectionPromise: Promise<Db> | null = null;

/**
 * Ensures a MongoDB connection exists. Safe to call on every request:
 * - Returns immediately if already connected.
 * - Reuses an in-flight connection attempt instead of starting a new one.
 * - On failure, clears the cached promise (so the NEXT request can retry)
 *   and rejects — it never calls process.exit and never crashes the
 *   Node.js process.
 */
const ensureDatabaseConnection = async (): Promise<Db> => {
  if (database) {
    return database;
  }

  if (!databaseConnectionPromise) {
    databaseConnectionPromise = connectDatabase()
      .then((db) => {
        database = db;
        return db;
      })
      .catch((error) => {
        databaseConnectionPromise = null;
        throw error;
      });
  }

  return databaseConnectionPromise;
};

/* =========================================================
   Authentication types
========================================================= */

type UserRole = "admin" | "doctor" | "patient";
type UserStatus = "active" | "blocked";

interface AuthenticatedRequest extends Request {
  userId?: string;
  userName?: string;
  userEmail?: string;
  userRole?: UserRole;
  userStatus?: UserStatus;
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

/**
 * Runs before every route. Makes sure MongoDB is connected before any
 * handler runs. If the connection fails, responds with a clean 503
 * instead of letting the process crash (the previous behavior).
 */
app.use(async (_req: Request, res: Response, next: NextFunction) => {
  try {
    await ensureDatabaseConnection();
    next();
  } catch (error) {
    console.error(
      "MongoDB connection error:",
      error instanceof Error ? error.message : error,
    );

    res.status(503).json({
      success: false,
      message: "Database is not connected",
    });
  }
});

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

      req.userStatus = currentUser.status === "blocked" ? "blocked" : "active";

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

/**
 * Allows blocked users to read data, but prevents them
 * from creating, editing, deleting, or changing status.
 *
 * Always use this after verifyRole().
 */
const verifyActive = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): void => {
  if (req.userStatus !== "active") {
    res.status(403).json({
      success: false,
      message:
        "Your account is blocked. You can view data, but you cannot perform this action.",
      code: "READ_ONLY_ACCOUNT",
    });

    return;
  }

  next();
};

/**
 * Allows any authenticated role (admin, doctor or patient)
 * to use protected features when the account status is active.
 */
const verifyAnyActiveUser = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): Promise<void> => {
  try {
    if (!req.userId) {
      res.status(401).json({
        success: false,
        message: "Authentication is required",
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

    const userQueryConditions: Record<string, unknown>[] = [
      {
        id: req.userId,
      },
    ];

    if (req.userEmail) {
      userQueryConditions.push({
        email: req.userEmail.toLowerCase(),
      });
    }

    const currentUser = await database.collection("user").findOne({
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

    const currentStatus: UserStatus =
      currentUser.status === "blocked" ? "blocked" : "active";

    req.userRole = currentRole as UserRole;
    req.userStatus = currentStatus;

    if (currentStatus !== "active") {
      res.status(403).json({
        success: false,
        message:
          "Your account is blocked. Only active accounts can use the AI Health Assistant.",
        code: "READ_ONLY_ACCOUNT",
      });

      return;
    }

    next();
  } catch (error) {
    console.error(
      "Active user verification error:",
      error instanceof Error ? error.message : error,
    );

    res.status(500).json({
      success: false,
      message: "Failed to verify active user account",
    });
  }
};

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
   Current authenticated user
========================================================= */

app.get(
  "/api/users/current",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      if (!req.userId) {
        res.status(401).json({
          success: false,
          message: "Authenticated user ID was not found",
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
          email: req.userEmail.toLowerCase(),
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

      const currentRole: UserRole =
        currentUser.role === "admin" ||
        currentUser.role === "doctor" ||
        currentUser.role === "patient"
          ? currentUser.role
          : "patient";

      const currentStatus: "active" | "blocked" =
        currentUser.status === "blocked" ? "blocked" : "active";

      const currentUserId =
        typeof currentUser.id === "string" && currentUser.id.trim()
          ? currentUser.id
          : currentUser._id instanceof ObjectId
            ? currentUser._id.toHexString()
            : req.userId;

      res.status(200).json({
        id: currentUserId,
        _id: currentUserId,
        name: typeof currentUser.name === "string" ? currentUser.name : null,
        email: typeof currentUser.email === "string" ? currentUser.email : null,
        image: typeof currentUser.image === "string" ? currentUser.image : null,
        role: currentRole,
        status: currentStatus,
      });
    } catch (error) {
      console.error(
        "Get current user error:",
        error instanceof Error ? error.message : error,
      );

      res.status(500).json({
        success: false,
        message: "Failed to retrieve current user",
      });
    }
  },
);

/* =========================================================
   Manage Doctors helpers
========================================================= */

type DoctorStatus = "active" | "blocked";

const getDoctorString = (value: unknown): string => {
  return typeof value === "string" ? value.trim() : "";
};

const getDoctorNumber = (value: unknown): number => {
  const numberValue =
    typeof value === "number"
      ? value
      : typeof value === "string" && value.trim()
        ? Number(value)
        : 0;

  return Number.isFinite(numberValue) ? Math.max(0, numberValue) : 0;
};

const normalizeDoctorEmail = (value: unknown): string => {
  return getDoctorString(value).toLowerCase();
};

const isValidDoctorEmail = (email: string): boolean => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
};

const escapeDoctorSearch = (value: string): string => {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
};

const getDoctorDocumentId = (document: Document): string => {
  if (typeof document.id === "string" && document.id.trim()) {
    return document.id;
  }

  if (document._id instanceof ObjectId) {
    return document._id.toHexString();
  }

  return String(document._id || "");
};

const getDoctorFilter = (doctorId: string): Filter<Document> => {
  const conditions: Filter<Document>[] = [
    {
      id: doctorId,
    },
  ];

  if (ObjectId.isValid(doctorId)) {
    conditions.push({
      _id: new ObjectId(doctorId),
    });
  }

  return {
    $or: conditions,
  };
};

const getUserFilter = (userId: string): Filter<Document> => {
  const conditions: Filter<Document>[] = [
    {
      id: userId,
    },
  ];

  if (ObjectId.isValid(userId)) {
    conditions.push({
      _id: new ObjectId(userId),
    });
  }

  return {
    $or: conditions,
  };
};

const formatDoctorDate = (value: unknown): string | null => {
  if (value instanceof Date) {
    return value.toISOString();
  }

  if (typeof value === "string" || typeof value === "number") {
    const date = new Date(value);

    return Number.isNaN(date.getTime()) ? null : date.toISOString();
  }

  return null;
};

const formatDoctor = (doctor: Document) => {
  return {
    id: getDoctorDocumentId(doctor),

    userId: typeof doctor.userId === "string" ? doctor.userId : "",

    name: getDoctorString(doctor.name),

    email: normalizeDoctorEmail(doctor.email),

    image: getDoctorString(doctor.image) || null,

    phone: getDoctorString(doctor.phone),

    specialization: getDoctorString(doctor.specialization),

    qualification: getDoctorString(doctor.qualification),

    experienceYears: getDoctorNumber(doctor.experienceYears),

    hospital:
      getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),

    address: getDoctorString(doctor.address),

    bio: getDoctorString(doctor.bio),

    role: "doctor" as const,

    status:
      doctor.status === "blocked" ? ("blocked" as const) : ("active" as const),

    createdAt: formatDoctorDate(doctor.createdAt),

    updatedAt: formatDoctorDate(doctor.updatedAt),
  };
};

const readBetterAuthResponse = async (
  response: globalThis.Response,
): Promise<unknown> => {
  try {
    return await response.json();
  } catch {
    return null;
  }
};

const getBetterAuthError = (value: unknown): string => {
  if (typeof value !== "object" || value === null) {
    return "Doctor authentication account could not be created";
  }

  const data = value as Record<string, unknown>;

  if (typeof data.message === "string" && data.message.trim()) {
    return data.message;
  }

  if (typeof data.error === "object" && data.error !== null) {
    const error = data.error as Record<string, unknown>;

    if (typeof error.message === "string" && error.message.trim()) {
      return error.message;
    }
  }

  return "Doctor authentication account could not be created";
};

/* =========================================================
   Admin patient management helpers
========================================================= */

type ManagedPatientStatus = "active" | "blocked";

const getAdminPatientFilter = (patientId: string): Filter<Document> => {
  return {
    $and: [getUserFilter(patientId), { role: "patient" }],
  };
};

const formatManagedPatient = (patient: Document) => {
  const status: ManagedPatientStatus =
    patient.status === "blocked" ? "blocked" : "active";

  return {
    id: getDoctorDocumentId(patient),
    name: getDoctorString(patient.name),
    email: normalizeDoctorEmail(patient.email),
    image: getDoctorString(patient.image) || null,
    role: "patient" as const,
    status,
    emailVerified: patient.emailVerified === true,
    phone: getDoctorString(patient.phone) || null,
    address: getDoctorString(patient.address) || null,
    dateOfBirth: getDoctorString(patient.dateOfBirth) || null,
    gender: getDoctorString(patient.gender) || null,
    bloodGroup: getDoctorString(patient.bloodGroup) || null,
    occupation: getDoctorString(patient.occupation) || null,
    city: getDoctorString(patient.city) || null,
    country: getDoctorString(patient.country) || null,
    bio: getDoctorString(patient.bio) || null,
    emergencyContactName: getDoctorString(patient.emergencyContactName) || null,
    emergencyContactPhone:
      getDoctorString(patient.emergencyContactPhone) ||
      getDoctorString(patient.emergencyContact) ||
      null,
    createdAt: formatDoctorDate(patient.createdAt),
    updatedAt: formatDoctorDate(patient.updatedAt),
  };
};

/* =========================================================
   GET managed patients (10 per page)
========================================================= */

app.get(
  "/api/v1/admin/patients",
  verifyToken,
  verifyAdmin,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const search = getDoctorString(req.query.search);
      const requestedStatus = getDoctorString(req.query.status);
      const requestedPage = getPositiveInteger(req.query.page, 1, 100000);
      const limit = 10;

      const conditions: Filter<Document>[] = [{ role: "patient" }];

      if (requestedStatus === "active" || requestedStatus === "blocked") {
        conditions.push({ status: requestedStatus });
      }

      if (search) {
        const safeSearch = escapeDoctorSearch(search);

        conditions.push({
          $or: [
            {
              name: {
                $regex: safeSearch,
                $options: "i",
              },
            },
            {
              email: {
                $regex: safeSearch,
                $options: "i",
              },
            },
          ],
        });
      }

      const filter: Filter<Document> = { $and: conditions };
      const usersCollection = database.collection("user");
      const total = await usersCollection.countDocuments(filter);
      const totalPages = Math.max(1, Math.ceil(total / limit));
      const page = Math.min(requestedPage, totalPages);

      const patientDocuments = await usersCollection
        .find(filter)
        .sort({
          updatedAt: -1,
          createdAt: -1,
          _id: -1,
        })
        .skip((page - 1) * limit)
        .limit(limit)
        .toArray();

      res.status(200).json({
        success: true,
        patients: patientDocuments.map(formatManagedPatient),
        pagination: {
          page,
          limit,
          total,
          totalPages,
        },
      });
    } catch (error) {
      console.error("Get managed patients error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve patients",
      });
    }
  },
);

/* =========================================================
   GET managed patient details
========================================================= */

app.get(
  "/api/v1/admin/patients/:patientId",
  verifyToken,
  verifyAdmin,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const patientId = getDoctorString(req.params.patientId);

      if (!patientId) {
        res.status(400).json({
          success: false,
          message: "Patient ID is required",
        });
        return;
      }

      const patient = await database
        .collection("user")
        .findOne(getAdminPatientFilter(patientId));

      if (!patient) {
        res.status(404).json({
          success: false,
          message: "Patient was not found",
        });
        return;
      }

      res.status(200).json({
        success: true,
        patient: formatManagedPatient(patient),
      });
    } catch (error) {
      console.error("Get managed patient details error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve patient details",
      });
    }
  },
);

/* =========================================================
   PATCH block or activate patient
========================================================= */

app.patch(
  "/api/v1/admin/patients/:patientId/status",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const patientId = getDoctorString(req.params.patientId);
      const requestedStatus = getDoctorString(req.body.status);

      if (requestedStatus !== "active" && requestedStatus !== "blocked") {
        res.status(400).json({
          success: false,
          message: "Status must be active or blocked",
        });
        return;
      }

      const usersCollection = database.collection("user");
      const patient = await usersCollection.findOne(
        getAdminPatientFilter(patientId),
      );

      if (!patient) {
        res.status(404).json({
          success: false,
          message: "Patient was not found",
        });
        return;
      }

      const status = requestedStatus as ManagedPatientStatus;
      const updatedPatient = await usersCollection.findOneAndUpdate(
        { _id: patient._id },
        {
          $set: {
            status,
            updatedAt: new Date(),
          },
        },
        { returnDocument: "after" },
      );

      if (!updatedPatient) {
        res.status(404).json({
          success: false,
          message: "Patient was not found",
        });
        return;
      }

      if (status === "blocked") {
        await database.collection("session").deleteMany({
          userId: getDoctorDocumentId(patient),
        });
      }

      res.status(200).json({
        success: true,
        message:
          status === "blocked"
            ? "Patient blocked successfully"
            : "Patient activated successfully",
        patient: formatManagedPatient(updatedPatient),
      });
    } catch (error) {
      console.error("Change patient status error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to change patient status",
      });
    }
  },
);

/* =========================================================
   DELETE patient account
========================================================= */

app.delete(
  "/api/v1/admin/patients/:patientId",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const patientId = getDoctorString(req.params.patientId);

      if (!patientId) {
        res.status(400).json({
          success: false,
          message: "Patient ID is required",
        });
        return;
      }

      const usersCollection = database.collection("user");
      const patient = await usersCollection.findOne(
        getAdminPatientFilter(patientId),
      );

      if (!patient) {
        res.status(404).json({
          success: false,
          message: "Patient was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(patient);
      const email = normalizeDoctorEmail(patient.email);

      await Promise.all([
        database.collection("session").deleteMany({ userId }),
        database.collection("account").deleteMany({ userId }),
        database.collection("verification").deleteMany({
          $or: [{ identifier: email }, { value: email }],
        }),
      ]);

      const deleteResult = await usersCollection.deleteOne({
        _id: patient._id,
      });

      if (deleteResult.deletedCount !== 1) {
        res.status(500).json({
          success: false,
          message: "Patient account could not be deleted",
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: "Patient account deleted successfully",
        deletedPatientId: userId,
      });
    } catch (error) {
      console.error("Delete patient account error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to delete patient account",
      });
    }
  },
);

/* =========================================================
   GET all doctors
========================================================= */

app.get(
  "/api/v1/admin/doctors",
  verifyToken,
  verifyAdmin,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorsCollection = database.collection("doctors");

      const search = getDoctorString(req.query.search);

      const status = getDoctorString(req.query.status);

      const page = Math.max(
        1,
        Math.floor(getDoctorNumber(req.query.page) || 1),
      );

      const limit = Math.min(
        100,
        Math.max(1, Math.floor(getDoctorNumber(req.query.limit) || 50)),
      );

      const filter: Filter<Document> = {};

      if (status === "active" || status === "blocked") {
        filter.status = status;
      }

      if (search) {
        const safeSearch = escapeDoctorSearch(search);

        filter.$or = [
          {
            name: {
              $regex: safeSearch,
              $options: "i",
            },
          },
          {
            email: {
              $regex: safeSearch,
              $options: "i",
            },
          },
          {
            phone: {
              $regex: safeSearch,
              $options: "i",
            },
          },
          {
            specialization: {
              $regex: safeSearch,
              $options: "i",
            },
          },
        ];
      }

      const [doctorDocuments, total] = await Promise.all([
        doctorsCollection
          .find(filter)
          .sort({
            createdAt: -1,
            _id: -1,
          })
          .skip((page - 1) * limit)
          .limit(limit)
          .toArray(),

        doctorsCollection.countDocuments(filter),
      ]);

      res.status(200).json({
        success: true,

        doctors: doctorDocuments.map(formatDoctor),

        pagination: {
          page,
          limit,
          total,

          totalPages: Math.max(1, Math.ceil(total / limit)),
        },
      });
    } catch (error) {
      console.error("Get doctors error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve doctors",
      });
    }
  },
);

/* =========================================================
   GET single doctor details
========================================================= */

app.get(
  "/api/v1/admin/doctors/:doctorId",
  verifyToken,
  verifyAdmin,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);

      if (!doctorId) {
        res.status(400).json({
          success: false,
          message: "Doctor ID is required",
        });

        return;
      }

      const doctorsCollection = database.collection("doctors");

      const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      res.status(200).json({
        success: true,
        doctor: formatDoctor(doctor),
      });
    } catch (error) {
      console.error("Get doctor details error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve doctor details",
      });
    }
  },
);

/* =========================================================
   POST create doctor
========================================================= */

app.post(
  "/api/v1/admin/doctors",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const name = getDoctorString(req.body.name);

      const email = normalizeDoctorEmail(req.body.email);

      const password = getDoctorString(req.body.password);

      const specialization = getDoctorString(req.body.specialization);

      if (!name || !email || !password || !specialization) {
        res.status(400).json({
          success: false,
          message: "Name, email, password and specialization are required",
        });

        return;
      }

      if (!isValidDoctorEmail(email)) {
        res.status(400).json({
          success: false,
          message: "A valid email address is required",
        });

        return;
      }

      if (password.length < 8) {
        res.status(400).json({
          success: false,
          message: "Password must contain at least 8 characters",
        });

        return;
      }

      const usersCollection = database.collection("user");

      const doctorsCollection = database.collection("doctors");

      const accountsCollection = database.collection("account");

      const sessionsCollection = database.collection("session");

      const existingUser = await usersCollection.findOne({
        email,
      });

      const existingDoctor = await doctorsCollection.findOne({
        email,
      });

      if (existingUser || existingDoctor) {
        res.status(409).json({
          success: false,
          message: "An account with this email already exists",
        });

        return;
      }

      /*
       * Better Auth securely creates the email/password account.
       * Raw password MongoDB-তে save হবে না।
       */
      const signupResponse = await fetch(
        `${betterAuthUrl}/api/auth/sign-up/email`,
        {
          method: "POST",

          headers: {
            "content-type": "application/json",
            accept: "application/json",
            origin: betterAuthUrl,
          },

          body: JSON.stringify({
            name,
            email,
            password,
          }),
        },
      );

      const signupData = await readBetterAuthResponse(signupResponse);

      if (!signupResponse.ok) {
        res
          .status(signupResponse.status >= 500 ? 502 : signupResponse.status)
          .json({
            success: false,
            message: getBetterAuthError(signupData),
          });

        return;
      }

      const createdUser = await usersCollection.findOne({
        email,
      });

      if (!createdUser) {
        res.status(500).json({
          success: false,
          message: "Authentication account was created but user was not found",
        });

        return;
      }

      const userId = getDoctorDocumentId(createdUser);

      const now = new Date();

      await usersCollection.updateOne(
        {
          _id: createdUser._id,
        },
        {
          $set: {
            name,
            email,
            role: "doctor",
            status: "active",
            updatedAt: now,
          },
        },
      );

      /*
       * Admin-created doctor will sign in manually.
       * Remove any session created by signup.
       */
      await sessionsCollection.deleteMany({
        userId,
      });

      const doctorDocument = {
        userId,

        name,
        email,

        image: getDoctorString(req.body.image) || null,

        phone: getDoctorString(req.body.phone),

        specialization,

        qualification: getDoctorString(req.body.qualification),

        experienceYears: getDoctorNumber(req.body.experienceYears),

        hospital: getDoctorString(req.body.hospital),

        address: getDoctorString(req.body.address),

        bio: getDoctorString(req.body.bio),

        role: "doctor" as const,

        status: "active" as const,

        createdAt: now,
        updatedAt: now,
      };

      try {
        const insertResult = await doctorsCollection.insertOne(doctorDocument);

        const createdDoctor = await doctorsCollection.findOne({
          _id: insertResult.insertedId,
        });

        if (!createdDoctor) {
          throw new Error("Created doctor profile was not found");
        }

        res.status(201).json({
          success: true,
          message: "Doctor created successfully",
          doctor: formatDoctor(createdDoctor),
        });
      } catch (profileError) {
        /*
         * Roll back authentication data if
         * doctor profile creation fails.
         */
        await Promise.all([
          sessionsCollection.deleteMany({
            userId,
          }),

          accountsCollection.deleteMany({
            userId,
          }),

          usersCollection.deleteOne({
            _id: createdUser._id,
          }),
        ]);

        throw profileError;
      }
    } catch (error) {
      console.error("Create doctor error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to create doctor",
      });
    }
  },
);

/* =========================================================
   PATCH edit doctor
========================================================= */

app.patch(
  "/api/v1/admin/doctors/:doctorId",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);

      const name = getDoctorString(req.body.name);

      const email = normalizeDoctorEmail(req.body.email);

      const specialization = getDoctorString(req.body.specialization);

      if (!doctorId || !name || !email || !specialization) {
        res.status(400).json({
          success: false,
          message: "Doctor ID, name, email and specialization are required",
        });

        return;
      }

      if (!isValidDoctorEmail(email)) {
        res.status(400).json({
          success: false,
          message: "A valid email address is required",
        });

        return;
      }

      const doctorsCollection = database.collection("doctors");

      const usersCollection = database.collection("user");

      const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      const userId = getDoctorString(doctor.userId);

      const linkedUser = userId
        ? await usersCollection.findOne(getUserFilter(userId))
        : null;

      const duplicateDoctor = await doctorsCollection.findOne({
        email,

        _id: {
          $ne: doctor._id,
        },
      });

      const duplicateUser = await usersCollection.findOne({
        email,

        ...(linkedUser
          ? {
              _id: {
                $ne: linkedUser._id,
              },
            }
          : {}),
      });

      if (duplicateDoctor || duplicateUser) {
        res.status(409).json({
          success: false,
          message: "Another account already uses this email",
        });

        return;
      }

      const now = new Date();

      const updatedDoctor = await doctorsCollection.findOneAndUpdate(
        {
          _id: doctor._id,
        },
        {
          $set: {
            name,
            email,

            image: getDoctorString(req.body.image) || null,

            phone: getDoctorString(req.body.phone),

            specialization,

            qualification: getDoctorString(req.body.qualification),

            experienceYears: getDoctorNumber(req.body.experienceYears),

            hospital: getDoctorString(req.body.hospital),

            address: getDoctorString(req.body.address),

            bio: getDoctorString(req.body.bio),

            updatedAt: now,
          },
        },
        {
          returnDocument: "after",
        },
      );

      if (!updatedDoctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      if (linkedUser) {
        await usersCollection.updateOne(
          {
            _id: linkedUser._id,
          },
          {
            $set: {
              name,
              email,

              image: getDoctorString(req.body.image) || null,

              updatedAt: now,
            },
          },
        );
      }

      res.status(200).json({
        success: true,
        message: "Doctor updated successfully",
        doctor: formatDoctor(updatedDoctor),
      });
    } catch (error) {
      console.error("Update doctor error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to update doctor",
      });
    }
  },
);

/* =========================================================
   PATCH block or activate doctor
========================================================= */

app.patch(
  "/api/v1/admin/doctors/:doctorId/status",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);

      const requestedStatus = getDoctorString(req.body.status);

      if (requestedStatus !== "active" && requestedStatus !== "blocked") {
        res.status(400).json({
          success: false,
          message: "Status must be active or blocked",
        });

        return;
      }

      const status = requestedStatus as DoctorStatus;

      const doctorsCollection = database.collection("doctors");

      const usersCollection = database.collection("user");

      const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      const now = new Date();

      const updatedDoctor = await doctorsCollection.findOneAndUpdate(
        {
          _id: doctor._id,
        },
        {
          $set: {
            status,
            updatedAt: now,
          },
        },
        {
          returnDocument: "after",
        },
      );

      if (!updatedDoctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      const userId = getDoctorString(doctor.userId);

      if (userId) {
        await usersCollection.updateOne(getUserFilter(userId), {
          $set: {
            status,
            updatedAt: now,
          },
        });
      }

      res.status(200).json({
        success: true,

        message:
          status === "blocked"
            ? "Doctor blocked successfully"
            : "Doctor activated successfully",

        doctor: formatDoctor(updatedDoctor),
      });
    } catch (error) {
      console.error("Change doctor status error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to change doctor status",
      });
    }
  },
);

/* =========================================================
   DELETE doctor
========================================================= */

app.delete(
  "/api/v1/admin/doctors/:doctorId",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);

      if (!doctorId) {
        res.status(400).json({
          success: false,
          message: "Doctor ID is required",
        });

        return;
      }

      const doctorsCollection = database.collection("doctors");

      const usersCollection = database.collection("user");

      const accountsCollection = database.collection("account");

      const sessionsCollection = database.collection("session");

      const verificationCollection = database.collection("verification");

      const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      const userId = getDoctorString(doctor.userId);

      if (userId) {
        await Promise.all([
          sessionsCollection.deleteMany({
            userId,
          }),

          accountsCollection.deleteMany({
            userId,
          }),

          verificationCollection.deleteMany({
            $or: [
              {
                identifier: doctor.email,
              },
              {
                value: doctor.email,
              },
            ],
          }),
        ]);

        await usersCollection.deleteOne(getUserFilter(userId));
      }

      const deleteResult = await doctorsCollection.deleteOne({
        _id: doctor._id,
      });

      if (deleteResult.deletedCount !== 1) {
        res.status(500).json({
          success: false,
          message: "Doctor could not be deleted",
        });

        return;
      }

      res.status(200).json({
        success: true,
        message: "Doctor deleted successfully",
        deletedDoctorId: getDoctorDocumentId(doctor),
      });
    } catch (error) {
      console.error("Delete doctor error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to delete doctor",
      });
    }
  },
);

/* =========================================================
   Public doctors, appointments and reviews
========================================================= */

type AppointmentStatus = "pending" | "approved" | "completed" | "rejected";

const ACTIVE_APPOINTMENT_STATUSES: AppointmentStatus[] = [
  "pending",
  "approved",
];

const getPositiveInteger = (
  value: unknown,
  fallback: number,
  maximum: number,
): number => {
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return Math.min(maximum, Math.max(1, Math.floor(parsed)));
};

const getCurrentDatabaseUser = async (
  req: AuthenticatedRequest,
): Promise<Document | null> => {
  if (!database || !req.userId) {
    return null;
  }

  const conditions: Filter<Document>[] = [getUserFilter(req.userId)];

  if (req.userEmail) {
    conditions.push({
      email: req.userEmail.toLowerCase(),
    });
  }

  return database.collection("user").findOne({
    $or: conditions,
  });
};

const getNormalizedUserRole = (user: Document): UserRole => {
  return user.role === "admin" ||
    user.role === "doctor" ||
    user.role === "patient"
    ? user.role
    : "patient";
};

const getNormalizedUserStatus = (user: Document): UserStatus => {
  return user.status === "blocked" ? "blocked" : "active";
};

const getPublicDoctor = (doctor: Document) => {
  const ratingAverage = Number(doctor.ratingAverage);

  const ratingCount = Number(doctor.ratingCount);

  return {
    ...formatDoctor(doctor),

    ratingAverage: Number.isFinite(ratingAverage)
      ? Number(ratingAverage.toFixed(1))
      : 0,

    ratingCount: Number.isFinite(ratingCount)
      ? Math.max(0, Math.floor(ratingCount))
      : 0,
  };
};

const getReviewDocumentId = (document: Document): string => {
  return getDoctorDocumentId(document);
};

const formatReview = (review: Document) => {
  return {
    id: getReviewDocumentId(review),
    doctorId: getDoctorString(review.doctorId),
    userId: getDoctorString(review.userId),
    userName: getDoctorString(review.userName),
    userEmail: normalizeDoctorEmail(review.userEmail),
    userImage: getDoctorString(review.userImage) || null,
    rating: Math.min(
      5,
      Math.max(1, Math.floor(getDoctorNumber(review.rating))),
    ),
    review: getDoctorString(review.review),
    createdAt: formatDoctorDate(review.createdAt),
    updatedAt: formatDoctorDate(review.updatedAt),
  };
};

const refreshDoctorRatingStats = async (doctorId: string): Promise<void> => {
  if (!database) {
    return;
  }

  const reviewsCollection = database.collection("reviews");

  const doctorsCollection = database.collection("doctors");

  const [stats] = await reviewsCollection
    .aggregate([
      {
        $match: {
          doctorId,
        },
      },
      {
        $group: {
          _id: "$doctorId",
          ratingAverage: {
            $avg: "$rating",
          },
          ratingCount: {
            $sum: 1,
          },
        },
      },
    ])
    .toArray();

  await doctorsCollection.updateOne(getDoctorFilter(doctorId), {
    $set: {
      ratingAverage:
        typeof stats?.ratingAverage === "number"
          ? Number(stats.ratingAverage.toFixed(2))
          : 0,
      ratingCount:
        typeof stats?.ratingCount === "number" ? stats.ratingCount : 0,
      updatedAt: new Date(),
    },
  });
};

const formatAppointment = (appointment: Document) => {
  return {
    id: getDoctorDocumentId(appointment),
    doctorId: getDoctorString(appointment.doctorId),
    doctorUserId: getDoctorString(appointment.doctorUserId),
    doctorName: getDoctorString(appointment.doctorName),
    doctorImage: getDoctorString(appointment.doctorImage) || null,
    specialization: getDoctorString(appointment.specialization),
    hospital: getDoctorString(appointment.hospital),
    patientUserId: getDoctorString(appointment.patientUserId),
    patientName: getDoctorString(appointment.patientName),
    patientEmail: normalizeDoctorEmail(appointment.patientEmail),
    patientImage: getDoctorString(appointment.patientImage) || null,
    phone: getDoctorString(appointment.phone),
    address: getDoctorString(appointment.address),
    problemTitle: getDoctorString(appointment.problemTitle),
    symptomsDescription: getDoctorString(appointment.symptomsDescription),
    appointmentDate: getDoctorString(appointment.appointmentDate),
    appointmentTime: getDoctorString(appointment.appointmentTime),
    status:
      appointment.status === "approved" ||
      appointment.status === "completed" ||
      appointment.status === "rejected"
        ? appointment.status
        : "pending",
    rejectionReason: getDoctorString(appointment.rejectionReason) || null,
    approvedAt: formatDoctorDate(appointment.approvedAt),
    completedAt: formatDoctorDate(appointment.completedAt),
    rejectedAt: formatDoctorDate(appointment.rejectedAt),
    rescheduledAt: formatDoctorDate(appointment.rescheduledAt),
    rescheduledBy: getDoctorString(appointment.rescheduledBy) || null,
    rescheduleReason: getDoctorString(appointment.rescheduleReason) || null,
    createdAt: formatDoctorDate(appointment.createdAt),
    updatedAt: formatDoctorDate(appointment.updatedAt),
  };
};

const attachPatientImages = async (
  appointments: Document[],
): Promise<Document[]> => {
  if (!database || appointments.length === 0) {
    return appointments;
  }

  const patientUserIds = Array.from(
    new Set(
      appointments
        .map((appointment) => getDoctorString(appointment.patientUserId))
        .filter(Boolean),
    ),
  );

  if (patientUserIds.length === 0) {
    return appointments;
  }

  const objectIds = patientUserIds
    .filter((userId) => ObjectId.isValid(userId))
    .map((userId) => new ObjectId(userId));

  const userConditions: Filter<Document>[] = [
    {
      id: {
        $in: patientUserIds,
      },
    },
  ];

  if (objectIds.length > 0) {
    userConditions.push({
      _id: {
        $in: objectIds,
      },
    });
  }

  const users = await database
    .collection("user")
    .find(
      {
        $or: userConditions,
      },
      {
        projection: {
          id: 1,
          image: 1,
        },
      },
    )
    .toArray();

  const imageByUserId = new Map<string, string | null>();

  users.forEach((user) => {
    imageByUserId.set(
      getDoctorDocumentId(user),
      getDoctorString(user.image) || null,
    );
  });

  return appointments.map((appointment) => ({
    ...appointment,
    patientImage:
      getDoctorString(appointment.patientImage) ||
      imageByUserId.get(getDoctorString(appointment.patientUserId)) ||
      null,
  }));
};

/* =========================================================
   Public doctor filters
========================================================= */

app.get(
  "/api/v1/doctors/filters",
  async (_req: Request, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorDocuments = await database
        .collection("doctors")
        .find(
          {
            status: "active",
          },
          {
            projection: {
              specialization: 1,
              qualification: 1,
              experienceYears: 1,
              hospital: 1,
              chamber: 1,
            },
          },
        )
        .toArray();

      const specializations = new Set<string>();
      const qualifications = new Set<string>();
      const hospitals = new Set<string>();
      const experienceYears = new Set<number>();

      doctorDocuments.forEach((doctor) => {
        const specialization = getDoctorString(doctor.specialization);
        const qualification = getDoctorString(doctor.qualification);
        const hospital =
          getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber);
        const experience = getDoctorNumber(doctor.experienceYears);

        if (specialization) {
          specializations.add(specialization);
        }

        if (qualification) {
          qualifications.add(qualification);
        }

        if (hospital) {
          hospitals.add(hospital);
        }

        experienceYears.add(experience);
      });

      res.status(200).json({
        success: true,
        filters: {
          specializations: Array.from(specializations).sort((a, b) =>
            a.localeCompare(b),
          ),
          qualifications: Array.from(qualifications).sort((a, b) =>
            a.localeCompare(b),
          ),
          hospitals: Array.from(hospitals).sort((a, b) => a.localeCompare(b)),
          experienceYears: Array.from(experienceYears).sort((a, b) => a - b),
        },
      });
    } catch (error) {
      console.error("Get public doctor filters error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve doctor filters",
      });
    }
  },
);

/* =========================================================
   Top Rated Public Doctors
========================================================= */

app.get(
  "/api/v1/doctors/top-rated",
  async (_req: Request, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorsCollection = database.collection("doctors");

      const doctors = await doctorsCollection
        .find({
          status: "active",
        })
        .sort({
          ratingAverage: -1,
          ratingCount: -1,
          createdAt: -1,
          _id: -1,
        })
        .limit(4)
        .toArray();

      res.status(200).json({
        success: true,

        doctors: doctors.map(getPublicDoctor),
      });
    } catch (error) {
      console.error("Get top rated doctors error:", error);

      res.status(500).json({
        success: false,

        message: "Failed to retrieve top rated doctors",
      });
    }
  },
);

/* =========================================================
   Public doctor list
========================================================= */

app.get(
  "/api/v1/doctors",
  async (req: Request, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const search = getDoctorString(req.query.search);
      const specialization = getDoctorString(req.query.specialization);
      const qualification = getDoctorString(req.query.qualification);
      const hospital = getDoctorString(req.query.hospital);
      const experienceValue = getDoctorString(req.query.experienceYears);

      const page = getPositiveInteger(req.query.page, 1, 100000);

      const limit = 8;

      const conditions: Filter<Document>[] = [
        {
          status: "active",
        },
      ];

      if (search) {
        const safeSearch = escapeDoctorSearch(search);

        conditions.push({
          $or: [
            {
              name: {
                $regex: safeSearch,
                $options: "i",
              },
            },
            {
              specialization: {
                $regex: safeSearch,
                $options: "i",
              },
            },
            {
              qualification: {
                $regex: safeSearch,
                $options: "i",
              },
            },
          ],
        });
      }

      if (specialization) {
        conditions.push({
          specialization: {
            $regex: `^${escapeDoctorSearch(specialization)}$`,
            $options: "i",
          },
        });
      }

      if (qualification) {
        conditions.push({
          qualification: {
            $regex: `^${escapeDoctorSearch(qualification)}$`,
            $options: "i",
          },
        });
      }

      if (hospital) {
        const safeHospital = `^${escapeDoctorSearch(hospital)}$`;

        conditions.push({
          $or: [
            {
              hospital: {
                $regex: safeHospital,
                $options: "i",
              },
            },
            {
              chamber: {
                $regex: safeHospital,
                $options: "i",
              },
            },
          ],
        });
      }

      if (experienceValue) {
        const experienceYears = Number(experienceValue);

        if (Number.isFinite(experienceYears)) {
          conditions.push({
            experienceYears: Math.max(0, Math.floor(experienceYears)),
          });
        }
      }

      const filter: Filter<Document> = {
        $and: conditions,
      };

      const doctorsCollection = database.collection("doctors");

      const [doctorDocuments, total] = await Promise.all([
        doctorsCollection
          .find(filter)
          .sort({
            ratingAverage: -1,
            ratingCount: -1,
            createdAt: -1,
            _id: -1,
          })
          .skip((page - 1) * limit)
          .limit(limit)
          .toArray(),

        doctorsCollection.countDocuments(filter),
      ]);

      res.status(200).json({
        success: true,
        doctors: doctorDocuments.map(getPublicDoctor),
        pagination: {
          page,
          limit,
          total,
          totalPages: Math.max(1, Math.ceil(total / limit)),
        },
      });
    } catch (error) {
      console.error("Get public doctors error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve public doctors",
      });
    }
  },
);

/* =========================================================
   Public single doctor details
========================================================= */

app.get(
  "/api/v1/doctors/:doctorId",
  async (req: Request, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);

      const doctor = await database.collection("doctors").findOne({
        $and: [
          getDoctorFilter(doctorId),
          {
            status: "active",
          },
        ],
      });

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      res.status(200).json({
        success: true,
        doctor: getPublicDoctor(doctor),
      });
    } catch (error) {
      console.error("Get public doctor details error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve doctor details",
      });
    }
  },
);

/* =========================================================
   Public doctor reviews
========================================================= */

app.get(
  "/api/v1/doctors/:doctorId/reviews",
  async (req: Request, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);
      const page = getPositiveInteger(req.query.page, 1, 100000);
      const limit = getPositiveInteger(req.query.limit, 10, 50);

      const reviewsCollection = database.collection("reviews");

      const [reviewDocuments, total] = await Promise.all([
        reviewsCollection
          .find({
            doctorId,
          })
          .sort({
            updatedAt: -1,
            createdAt: -1,
          })
          .skip((page - 1) * limit)
          .limit(limit)
          .toArray(),

        reviewsCollection.countDocuments({
          doctorId,
        }),
      ]);

      res.status(200).json({
        success: true,
        reviews: reviewDocuments.map(formatReview),
        pagination: {
          page,
          limit,
          total,
          totalPages: Math.max(1, Math.ceil(total / limit)),
        },
      });
    } catch (error) {
      console.error("Get doctor reviews error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve doctor reviews",
      });
    }
  },
);

/* =========================================================
   Create doctor review
========================================================= */

app.post(
  "/api/v1/doctors/:doctorId/reviews",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      if (getNormalizedUserStatus(currentUser) === "blocked") {
        res.status(403).json({
          success: false,
          message:
            "You are restricted by the administrator and cannot submit a rating or review.",
          code: "ACCOUNT_BLOCKED",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);
      const rating = Math.floor(Number(req.body.rating));
      const reviewText = getDoctorString(req.body.review);

      if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
        res.status(400).json({
          success: false,
          message: "Rating must be a number from 1 to 5",
        });

        return;
      }

      if (reviewText.length > 2000) {
        res.status(400).json({
          success: false,
          message: "Review cannot contain more than 2000 characters",
        });

        return;
      }

      const doctor = await database.collection("doctors").findOne({
        $and: [
          getDoctorFilter(doctorId),
          {
            status: "active",
          },
        ],
      });

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      const currentUserId = getDoctorDocumentId(currentUser);

      if (getDoctorString(doctor.userId) === currentUserId) {
        res.status(403).json({
          success: false,
          message: "A doctor cannot review their own profile",
        });

        return;
      }

      const reviewsCollection = database.collection("reviews");

      const existingReview = await reviewsCollection.findOne({
        doctorId,
        userId: currentUserId,
      });

      if (existingReview) {
        res.status(409).json({
          success: false,
          message:
            "You have already reviewed this doctor. Please edit your existing review.",
          code: "REVIEW_ALREADY_EXISTS",
        });

        return;
      }

      const now = new Date();

      const reviewDocument = {
        doctorId,
        doctorUserId: getDoctorString(doctor.userId),
        userId: currentUserId,
        userName: getDoctorString(currentUser.name),
        userEmail: normalizeDoctorEmail(currentUser.email),
        userImage: getDoctorString(currentUser.image) || null,
        rating,
        review: reviewText,
        createdAt: now,
        updatedAt: now,
      };

      const insertResult = await reviewsCollection.insertOne(reviewDocument);

      await refreshDoctorRatingStats(doctorId);

      const createdReview = await reviewsCollection.findOne({
        _id: insertResult.insertedId,
      });

      res.status(201).json({
        success: true,
        message: "Rating and review submitted successfully",
        review: createdReview ? formatReview(createdReview) : null,
      });
    } catch (error) {
      console.error("Create doctor review error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to submit rating and review",
      });
    }
  },
);

/* =========================================================
   Update doctor review
========================================================= */

app.patch(
  "/api/v1/doctors/:doctorId/reviews/:reviewId",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      if (getNormalizedUserStatus(currentUser) === "blocked") {
        res.status(403).json({
          success: false,
          message:
            "You are restricted by the administrator and cannot edit a review.",
          code: "ACCOUNT_BLOCKED",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);
      const reviewId = getDoctorString(req.params.reviewId);
      const rating = Math.floor(Number(req.body.rating));
      const reviewText = getDoctorString(req.body.review);

      if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
        res.status(400).json({
          success: false,
          message: "Rating must be a number from 1 to 5",
        });

        return;
      }

      if (reviewText.length > 2000) {
        res.status(400).json({
          success: false,
          message: "Review cannot contain more than 2000 characters",
        });

        return;
      }

      const reviewsCollection = database.collection("reviews");

      const existingReview = await reviewsCollection.findOne({
        $and: [
          getDoctorFilter(reviewId),
          {
            doctorId,
          },
        ],
      });

      if (!existingReview) {
        res.status(404).json({
          success: false,
          message: "Review was not found",
        });

        return;
      }

      const currentUserId = getDoctorDocumentId(currentUser);

      if (getDoctorString(existingReview.userId) !== currentUserId) {
        res.status(403).json({
          success: false,
          message: "You can edit only your own review",
        });

        return;
      }

      const updatedReview = await reviewsCollection.findOneAndUpdate(
        {
          _id: existingReview._id,
        },
        {
          $set: {
            rating,
            review: reviewText,
            updatedAt: new Date(),
          },
        },
        {
          returnDocument: "after",
        },
      );

      await refreshDoctorRatingStats(doctorId);

      res.status(200).json({
        success: true,
        message: "Rating and review updated successfully",
        review: updatedReview ? formatReview(updatedReview) : null,
      });
    } catch (error) {
      console.error("Update doctor review error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to update rating and review",
      });
    }
  },
);

/* =========================================================
   Delete doctor review
========================================================= */

app.delete(
  "/api/v1/doctors/:doctorId/reviews/:reviewId",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      if (getNormalizedUserStatus(currentUser) === "blocked") {
        res.status(403).json({
          success: false,
          message:
            "You are restricted by the administrator and cannot delete a review.",
          code: "ACCOUNT_BLOCKED",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);
      const reviewId = getDoctorString(req.params.reviewId);

      const reviewsCollection = database.collection("reviews");

      const existingReview = await reviewsCollection.findOne({
        $and: [
          getDoctorFilter(reviewId),
          {
            doctorId,
          },
        ],
      });

      if (!existingReview) {
        res.status(404).json({
          success: false,
          message: "Review was not found",
        });

        return;
      }

      const currentUserId = getDoctorDocumentId(currentUser);

      if (getDoctorString(existingReview.userId) !== currentUserId) {
        res.status(403).json({
          success: false,
          message: "You can delete only your own review",
        });

        return;
      }

      await reviewsCollection.deleteOne({
        _id: existingReview._id,
      });

      await refreshDoctorRatingStats(doctorId);

      res.status(200).json({
        success: true,
        message: "Review deleted successfully",
        deletedReviewId: reviewId,
      });
    } catch (error) {
      console.error("Delete doctor review error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to delete review",
      });
    }
  },
);

/* =========================================================
   Appointment eligibility
========================================================= */

app.get(
  "/api/v1/appointments/eligibility/:doctorId",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const role = getNormalizedUserRole(currentUser);
      const status = getNormalizedUserStatus(currentUser);

      if (role !== "patient") {
        res.status(403).json({
          success: false,
          canBook: false,
          code: "PATIENT_ONLY",
          message: "Only patients can take a doctor appointment.",
        });

        return;
      }

      if (status === "blocked") {
        res.status(403).json({
          success: false,
          canBook: false,
          code: "ACCOUNT_BLOCKED",
          message:
            "You are restricted by the administrator and cannot take an appointment.",
        });

        return;
      }

      const doctorId = getDoctorString(req.params.doctorId);

      const doctor = await database.collection("doctors").findOne({
        $and: [
          getDoctorFilter(doctorId),
          {
            status: "active",
          },
        ],
      });

      if (!doctor) {
        res.status(404).json({
          success: false,
          canBook: false,
          message: "Doctor was not found",
        });

        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);

      const existingAppointment = await database
        .collection("appointments")
        .findOne({
          doctorId,
          patientUserId,
          status: {
            $in: ACTIVE_APPOINTMENT_STATUSES,
          },
        });

      if (existingAppointment) {
        res.status(200).json({
          success: true,
          canBook: false,
          code: "APPOINTMENT_ALREADY_EXISTS",
          message:
            "You already have a pending or approved appointment with this doctor.",
          appointment: formatAppointment(existingAppointment),
        });

        return;
      }

      res.status(200).json({
        success: true,
        canBook: true,
        message: "You can take an appointment with this doctor.",
      });
    } catch (error) {
      console.error("Appointment eligibility error:", error);

      res.status(500).json({
        success: false,
        canBook: false,
        message: "Failed to check appointment eligibility",
      });
    }
  },
);

/* =========================================================
   Create appointment
========================================================= */

app.post(
  "/api/v1/appointments",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const role = getNormalizedUserRole(currentUser);
      const status = getNormalizedUserStatus(currentUser);

      if (role !== "patient") {
        res.status(403).json({
          success: false,
          message: "Only patients can take a doctor appointment.",
          code: "PATIENT_ONLY",
        });

        return;
      }

      if (status === "blocked") {
        res.status(403).json({
          success: false,
          message:
            "You are restricted by the administrator and cannot take an appointment.",
          code: "ACCOUNT_BLOCKED",
        });

        return;
      }

      const doctorId = getDoctorString(req.body.doctorId);
      const patientName = getDoctorString(req.body.patientName);
      const phone = getDoctorString(req.body.phone);
      const address = getDoctorString(req.body.address);
      const problemTitle = getDoctorString(req.body.problemTitle);
      const symptomsDescription = getDoctorString(req.body.symptomsDescription);
      const appointmentDate = getDoctorString(req.body.appointmentDate);
      const appointmentTime = getDoctorString(req.body.appointmentTime);

      if (
        !doctorId ||
        !patientName ||
        !phone ||
        !address ||
        !problemTitle ||
        !symptomsDescription ||
        !appointmentDate ||
        !appointmentTime
      ) {
        res.status(400).json({
          success: false,
          message:
            "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
        });

        return;
      }

      if (symptomsDescription.length > 5000) {
        res.status(400).json({
          success: false,
          message:
            "Symptoms description cannot contain more than 5000 characters",
        });

        return;
      }

      const datePattern = /^\d{4}-\d{2}-\d{2}$/;
      const timePattern = /^\d{2}:\d{2}$/;

      if (
        !datePattern.test(appointmentDate) ||
        !timePattern.test(appointmentTime)
      ) {
        res.status(400).json({
          success: false,
          message: "A valid appointment date and time are required",
        });

        return;
      }

      const today = new Date().toISOString().slice(0, 10);

      if (appointmentDate < today) {
        res.status(400).json({
          success: false,
          message: "Appointment date cannot be in the past",
        });

        return;
      }

      const doctor = await database.collection("doctors").findOne({
        $and: [
          getDoctorFilter(doctorId),
          {
            status: "active",
          },
        ],
      });

      if (!doctor) {
        res.status(404).json({
          success: false,
          message: "Doctor was not found",
        });

        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const appointmentsCollection = database.collection("appointments");

      const existingAppointment = await appointmentsCollection.findOne({
        doctorId,
        patientUserId,
        status: {
          $in: ACTIVE_APPOINTMENT_STATUSES,
        },
      });

      if (existingAppointment) {
        res.status(409).json({
          success: false,
          message:
            "You already have a pending or approved appointment with this doctor.",
          code: "APPOINTMENT_ALREADY_EXISTS",
          appointment: formatAppointment(existingAppointment),
        });

        return;
      }

      const now = new Date();

      const appointmentDocument = {
        doctorId,
        doctorUserId: getDoctorString(doctor.userId),
        doctorName: getDoctorString(doctor.name),
        doctorImage: getDoctorString(doctor.image) || null,
        specialization: getDoctorString(doctor.specialization),
        hospital:
          getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
        patientUserId,
        patientName,
        patientEmail: normalizeDoctorEmail(currentUser.email),
        patientImage: getDoctorString(currentUser.image) || null,
        phone,
        address,
        problemTitle,
        symptomsDescription,
        appointmentDate,
        appointmentTime,
        status: "pending" as const,
        createdAt: now,
        updatedAt: now,
      };

      const insertResult =
        await appointmentsCollection.insertOne(appointmentDocument);

      const createdAppointment = await appointmentsCollection.findOne({
        _id: insertResult.insertedId,
      });

      res.status(201).json({
        success: true,
        message: "Appointment request submitted successfully",
        appointment: createdAppointment
          ? formatAppointment(createdAppointment)
          : null,
      });
    } catch (error) {
      console.error("Create appointment error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to submit appointment request",
      });
    }
  },
);

/* =========================================================
   Patient appointment helpers
========================================================= */

const getPatientAppointment = async (
  patientUserId: string,
  appointmentId: string,
): Promise<Document | null> => {
  if (!database) {
    return null;
  }

  return database.collection("appointments").findOne({
    $and: [
      getDoctorFilter(appointmentId),
      {
        patientUserId,
      },
    ],
  });
};

const getAppointmentDoctor = async (
  appointment: Document,
): Promise<Document | null> => {
  if (!database) {
    return null;
  }

  const doctorId = getDoctorString(appointment.doctorId);

  if (!doctorId) {
    return null;
  }

  return database.collection("doctors").findOne(getDoctorFilter(doctorId));
};

const validatePatientAppointmentInput = (
  body: Record<string, unknown>,
):
  | {
      success: true;
      values: {
        patientName: string;
        phone: string;
        address: string;
        problemTitle: string;
        symptomsDescription: string;
        appointmentDate: string;
        appointmentTime: string;
      };
    }
  | {
      success: false;
      message: string;
    } => {
  const patientName = getDoctorString(body.patientName);
  const phone = getDoctorString(body.phone);
  const address = getDoctorString(body.address);
  const problemTitle = getDoctorString(body.problemTitle);
  const symptomsDescription = getDoctorString(body.symptomsDescription);
  const appointmentDate = getDoctorString(body.appointmentDate);
  const appointmentTime = getDoctorString(body.appointmentTime);

  if (
    !patientName ||
    !phone ||
    !address ||
    !problemTitle ||
    !symptomsDescription ||
    !appointmentDate ||
    !appointmentTime
  ) {
    return {
      success: false,
      message:
        "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
    };
  }

  if (patientName.length > 150) {
    return {
      success: false,
      message: "Patient name cannot contain more than 150 characters",
    };
  }

  if (phone.length > 40) {
    return {
      success: false,
      message: "Phone number cannot contain more than 40 characters",
    };
  }

  if (address.length > 500) {
    return {
      success: false,
      message: "Address cannot contain more than 500 characters",
    };
  }

  if (problemTitle.length > 250) {
    return {
      success: false,
      message: "Health problem title cannot contain more than 250 characters",
    };
  }

  if (symptomsDescription.length > 5000) {
    return {
      success: false,
      message: "Symptoms description cannot contain more than 5000 characters",
    };
  }

  const datePattern = /^\d{4}-\d{2}-\d{2}$/;
  const timePattern = /^\d{2}:\d{2}$/;

  if (
    !datePattern.test(appointmentDate) ||
    !timePattern.test(appointmentTime)
  ) {
    return {
      success: false,
      message: "A valid appointment date and time are required",
    };
  }

  const today = new Date().toISOString().slice(0, 10);

  if (appointmentDate < today) {
    return {
      success: false,
      message: "Appointment date cannot be in the past",
    };
  }

  return {
    success: true,
    values: {
      patientName,
      phone,
      address,
      problemTitle,
      symptomsDescription,
      appointmentDate,
      appointmentTime,
    },
  };
};

/* =========================================================
   Patient appointments list
========================================================= */

app.get(
  "/api/v1/patient/appointments",
  verifyToken,
  verifyPatient,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const page = getPositiveInteger(req.query.page, 1, 100000);
      const limit = 10;
      const appointmentsCollection = database.collection("appointments");
      const filter: Filter<Document> = { patientUserId };

      const [appointmentDocuments, total] = await Promise.all([
        appointmentsCollection
          .find(filter)
          .sort({ createdAt: -1, _id: -1 })
          .skip((page - 1) * limit)
          .limit(limit)
          .toArray(),
        appointmentsCollection.countDocuments(filter),
      ]);

      res.status(200).json({
        success: true,
        appointments: appointmentDocuments.map(formatAppointment),
        pagination: {
          page,
          limit,
          total,
          totalPages: Math.max(1, Math.ceil(total / limit)),
        },
      });
    } catch (error) {
      console.error("Get patient appointments error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to retrieve appointments",
      });
    }
  },
);

/* =========================================================
   Patient single appointment details
========================================================= */

app.get(
  "/api/v1/patient/appointments/:appointmentId",
  verifyToken,
  verifyPatient,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const appointmentId = getDoctorString(req.params.appointmentId);

      if (!appointmentId) {
        res.status(400).json({
          success: false,
          message: "Appointment ID is required",
        });
        return;
      }

      const appointment = await getPatientAppointment(
        patientUserId,
        appointmentId,
      );

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });
        return;
      }

      const doctor = await getAppointmentDoctor(appointment);

      res.status(200).json({
        success: true,
        appointment: formatAppointment(appointment),
        doctor: doctor ? getPublicDoctor(doctor) : null,
      });
    } catch (error) {
      console.error("Get patient appointment details error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to retrieve appointment details",
      });
    }
  },
);

/* =========================================================
   Patient edit appointment
========================================================= */

app.patch(
  "/api/v1/patient/appointments/:appointmentId",
  verifyToken,
  verifyPatient,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const appointmentId = getDoctorString(req.params.appointmentId);
      const appointment = await getPatientAppointment(
        patientUserId,
        appointmentId,
      );

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });
        return;
      }

      const currentStatus = getDoctorString(appointment.status);

      if (currentStatus !== "pending" && currentStatus !== "rejected") {
        res.status(409).json({
          success: false,
          message: "Only pending or rejected appointments can be edited",
        });
        return;
      }

      const validation = validatePatientAppointmentInput(
        req.body as Record<string, unknown>,
      );

      if (!validation.success) {
        res.status(400).json({
          success: false,
          message: validation.message,
        });
        return;
      }

      if (currentStatus === "rejected") {
        const anotherActiveAppointment = await database
          .collection("appointments")
          .findOne({
            _id: { $ne: appointment._id },
            doctorId: getDoctorString(appointment.doctorId),
            patientUserId,
            status: { $in: ACTIVE_APPOINTMENT_STATUSES },
          });

        if (anotherActiveAppointment) {
          res.status(409).json({
            success: false,
            message:
              "You already have another pending or approved appointment with this doctor.",
          });
          return;
        }
      }

      const now = new Date();
      const updatedAppointment = await database
        .collection("appointments")
        .findOneAndUpdate(
          { _id: appointment._id },
          {
            $set: {
              ...validation.values,
              status: "pending",
              rejectionReason: null,
              rejectedAt: null,
              approvedAt: null,
              completedAt: null,
              updatedAt: now,
            },
          },
          { returnDocument: "after" },
        );

      res.status(200).json({
        success: true,
        message:
          currentStatus === "rejected"
            ? "Appointment updated and resubmitted successfully"
            : "Appointment updated successfully",
        appointment: updatedAppointment
          ? formatAppointment(updatedAppointment)
          : null,
      });
    } catch (error) {
      console.error("Update patient appointment error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to update appointment",
      });
    }
  },
);

/* =========================================================
   Patient cancel and delete appointment
========================================================= */

app.delete(
  "/api/v1/patient/appointments/:appointmentId",
  verifyToken,
  verifyPatient,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const appointmentId = getDoctorString(req.params.appointmentId);
      const appointment = await getPatientAppointment(
        patientUserId,
        appointmentId,
      );

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });
        return;
      }

      if (getDoctorString(appointment.status) === "completed") {
        res.status(409).json({
          success: false,
          message: "A completed appointment cannot be cancelled or deleted",
        });
        return;
      }

      const deleteResult = await database
        .collection("appointments")
        .deleteOne({ _id: appointment._id });

      if (deleteResult.deletedCount !== 1) {
        res.status(500).json({
          success: false,
          message: "Appointment could not be cancelled",
        });
        return;
      }

      res.status(200).json({
        success: true,
        message: "Appointment cancelled and removed successfully",
        deletedAppointmentId: appointmentId,
      });
    } catch (error) {
      console.error("Cancel patient appointment error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to cancel appointment",
      });
    }
  },
);

const getAppointmentListFilter = (
  req: AuthenticatedRequest,
): Filter<Document> => {
  const conditions: Filter<Document>[] = [];
  const status = getDoctorString(req.query.status);
  const search = getDoctorString(req.query.search);

  if (
    status === "pending" ||
    status === "approved" ||
    status === "completed" ||
    status === "rejected"
  ) {
    conditions.push({
      status,
    });
  }

  if (search) {
    const safeSearch = escapeDoctorSearch(search);

    conditions.push({
      $or: [
        {
          patientName: {
            $regex: safeSearch,
            $options: "i",
          },
        },
        {
          patientEmail: {
            $regex: safeSearch,
            $options: "i",
          },
        },
        {
          doctorName: {
            $regex: safeSearch,
            $options: "i",
          },
        },
        {
          problemTitle: {
            $regex: safeSearch,
            $options: "i",
          },
        },
      ],
    });
  }

  return conditions.length
    ? {
        $and: conditions,
      }
    : {};
};

const sendAppointmentList = async (
  req: AuthenticatedRequest,
  res: Response,
  additionalFilter: Filter<Document> = {},
): Promise<void> => {
  if (!database) {
    res.status(503).json({
      success: false,
      message: "Database is not connected",
    });

    return;
  }

  const page = getPositiveInteger(req.query.page, 1, 100000);
  const limit = 10;

  const queryFilter = getAppointmentListFilter(req);

  const filter: Filter<Document> = {
    $and: [queryFilter, additionalFilter],
  };

  const appointmentsCollection = database.collection("appointments");

  const [appointmentDocuments, total] = await Promise.all([
    appointmentsCollection
      .find(filter)
      .sort({
        createdAt: -1,
        _id: -1,
      })
      .skip((page - 1) * limit)
      .limit(limit)
      .toArray(),

    appointmentsCollection.countDocuments(filter),
  ]);

  const appointmentsWithImages =
    await attachPatientImages(appointmentDocuments);

  res.status(200).json({
    success: true,
    appointments: appointmentsWithImages.map(formatAppointment),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.max(1, Math.ceil(total / limit)),
    },
  });
};

/* =========================================================
   Admin appointment management
========================================================= */

app.get(
  "/api/v1/admin/appointments",
  verifyToken,
  verifyAdmin,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      await sendAppointmentList(req, res);
    } catch (error) {
      console.error("Get admin appointments error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve appointments",
      });
    }
  },
);

/* =========================================================
   Doctor appointment management
========================================================= */

app.get(
  "/api/v1/doctor/appointments",
  verifyToken,
  verifyDoctor,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const doctorUserId = getDoctorDocumentId(currentUser);

      await sendAppointmentList(req, res, {
        doctorUserId,
      });
    } catch (error) {
      console.error("Get doctor appointments error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve appointments",
      });
    }
  },
);

/* =========================================================
   Doctor single appointment details
========================================================= */

app.get(
  "/api/v1/doctor/appointments/:appointmentId",
  verifyToken,
  verifyDoctor,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const appointmentId = getDoctorString(req.params.appointmentId);

      if (!appointmentId) {
        res.status(400).json({
          success: false,
          message: "Appointment ID is required",
        });

        return;
      }

      const doctorUserId = getDoctorDocumentId(currentUser);

      const appointment = await database.collection("appointments").findOne({
        $and: [
          getDoctorFilter(appointmentId),
          {
            doctorUserId,
          },
        ],
      });

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });

        return;
      }

      const [appointmentWithImage] = await attachPatientImages([appointment]);

      res.status(200).json({
        success: true,
        appointment: formatAppointment(appointmentWithImage),
      });
    } catch (error) {
      console.error("Get doctor appointment details error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve appointment details",
      });
    }
  },
);

/* =========================================================
   Doctor appointment reschedule
========================================================= */

app.patch(
  "/api/v1/doctor/appointments/:appointmentId/reschedule",
  verifyToken,
  verifyDoctor,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const appointmentId = getDoctorString(req.params.appointmentId);

      const appointmentDate = getDoctorString(req.body.appointmentDate);

      const appointmentTime = getDoctorString(req.body.appointmentTime);

      const rescheduleReason = getDoctorString(req.body.rescheduleReason);

      const datePattern = /^\d{4}-\d{2}-\d{2}$/;
      const timePattern = /^\d{2}:\d{2}$/;

      if (
        !datePattern.test(appointmentDate) ||
        !timePattern.test(appointmentTime)
      ) {
        res.status(400).json({
          success: false,
          message: "A valid appointment date and time are required",
        });

        return;
      }

      const today = new Date().toISOString().slice(0, 10);

      if (appointmentDate < today) {
        res.status(400).json({
          success: false,
          message: "Appointment date cannot be in the past",
        });

        return;
      }

      if (rescheduleReason.length > 1000) {
        res.status(400).json({
          success: false,
          message: "Reschedule reason cannot contain more than 1000 characters",
        });

        return;
      }

      const appointmentsCollection = database.collection("appointments");

      const appointment = await appointmentsCollection.findOne(
        getDoctorFilter(appointmentId),
      );

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });

        return;
      }

      const doctorUserId = getDoctorDocumentId(currentUser);

      if (getDoctorString(appointment.doctorUserId) !== doctorUserId) {
        res.status(403).json({
          success: false,
          message: "You can reschedule only your own appointments",
        });

        return;
      }

      const currentStatus = getDoctorString(appointment.status);

      if (
        currentStatus !== "pending" &&
        currentStatus !== "approved" &&
        currentStatus !== "rejected"
      ) {
        res.status(409).json({
          success: false,
          message:
            "Only pending, approved or rejected appointments can be rescheduled",
        });

        return;
      }

      const now = new Date();

      const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
        {
          _id: appointment._id,
        },
        {
          $set: {
            appointmentDate,
            appointmentTime,
            rescheduleReason: rescheduleReason || null,
            rescheduledAt: now,
            rescheduledBy: doctorUserId,
            updatedAt: now,
          },
        },
        {
          returnDocument: "after",
        },
      );

      res.status(200).json({
        success: true,
        message: "Appointment rescheduled successfully",
        appointment: updatedAppointment
          ? formatAppointment(updatedAppointment)
          : null,
      });
    } catch (error) {
      console.error("Doctor reschedule appointment error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to reschedule appointment",
      });
    }
  },
);

/* =========================================================
   Admin-only appointment reschedule
========================================================= */

app.patch(
  "/api/v1/admin/appointments/:appointmentId/reschedule",
  verifyToken,
  verifyAdmin,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const appointmentId = getDoctorString(req.params.appointmentId);

      const appointmentDate = getDoctorString(req.body.appointmentDate);

      const appointmentTime = getDoctorString(req.body.appointmentTime);

      const rescheduleReason = getDoctorString(req.body.rescheduleReason);

      const datePattern = /^\d{4}-\d{2}-\d{2}$/;
      const timePattern = /^\d{2}:\d{2}$/;

      if (
        !datePattern.test(appointmentDate) ||
        !timePattern.test(appointmentTime)
      ) {
        res.status(400).json({
          success: false,
          message: "A valid appointment date and time are required",
        });

        return;
      }

      const today = new Date().toISOString().slice(0, 10);

      if (appointmentDate < today) {
        res.status(400).json({
          success: false,
          message: "Appointment date cannot be in the past",
        });

        return;
      }

      if (rescheduleReason.length > 1000) {
        res.status(400).json({
          success: false,
          message: "Reschedule reason cannot contain more than 1000 characters",
        });

        return;
      }

      const appointmentsCollection = database.collection("appointments");

      const appointment = await appointmentsCollection.findOne(
        getDoctorFilter(appointmentId),
      );

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });

        return;
      }

      const currentStatus = getDoctorString(appointment.status);

      if (currentStatus === "completed" || currentStatus === "rejected") {
        res.status(409).json({
          success: false,
          message: "A completed or rejected appointment cannot be rescheduled",
        });

        return;
      }

      const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
        {
          _id: appointment._id,
        },
        {
          $set: {
            appointmentDate,
            appointmentTime,
            rescheduleReason: rescheduleReason || null,
            rescheduledAt: new Date(),
            rescheduledBy: req.userId || null,
            updatedAt: new Date(),
          },
        },
        {
          returnDocument: "after",
        },
      );

      res.status(200).json({
        success: true,
        message: "Appointment rescheduled successfully",
        appointment: updatedAppointment
          ? formatAppointment(updatedAppointment)
          : null,
      });
    } catch (error) {
      console.error("Reschedule appointment error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to reschedule appointment",
      });
    }
  },
);

/* =========================================================
   Admin or doctor appointment status update
========================================================= */

app.patch(
  "/api/v1/appointments/:appointmentId/status",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });

        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });

        return;
      }

      const role = getNormalizedUserRole(currentUser);
      const userStatus = getNormalizedUserStatus(currentUser);

      if (role !== "admin" && role !== "doctor") {
        res.status(403).json({
          success: false,
          message:
            "Only an administrator or doctor can update appointment status.",
        });

        return;
      }

      if (userStatus === "blocked") {
        res.status(403).json({
          success: false,
          message:
            "Your account is blocked. You can view appointments but cannot update them.",
          code: "READ_ONLY_ACCOUNT",
        });

        return;
      }

      const requestedStatus = getDoctorString(req.body.status);
      const rejectionReason = getDoctorString(req.body.rejectionReason);

      if (
        requestedStatus !== "approved" &&
        requestedStatus !== "completed" &&
        requestedStatus !== "rejected"
      ) {
        res.status(400).json({
          success: false,
          message: "Status must be approved, completed or rejected",
        });

        return;
      }

      if (requestedStatus === "rejected" && !rejectionReason) {
        res.status(400).json({
          success: false,
          message:
            "A rejection message is required when rejecting an appointment",
        });

        return;
      }

      if (rejectionReason.length > 1000) {
        res.status(400).json({
          success: false,
          message: "Rejection message cannot contain more than 1000 characters",
        });

        return;
      }

      const appointmentId = getDoctorString(req.params.appointmentId);

      const appointmentsCollection = database.collection("appointments");

      const appointment = await appointmentsCollection.findOne(
        getDoctorFilter(appointmentId),
      );

      if (!appointment) {
        res.status(404).json({
          success: false,
          message: "Appointment was not found",
        });

        return;
      }

      if (role === "doctor") {
        const currentUserId = getDoctorDocumentId(currentUser);

        if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
          res.status(403).json({
            success: false,
            message: "You can update only your own appointments",
          });

          return;
        }
      }

      const currentStatus = getDoctorString(appointment.status);

      if (currentStatus === "completed") {
        res.status(409).json({
          success: false,
          message: "A completed appointment cannot be changed",
        });

        return;
      }

      if (requestedStatus === "completed" && currentStatus !== "approved") {
        res.status(409).json({
          success: false,
          message: "Only an approved appointment can be marked as completed",
        });

        return;
      }

      if (
        requestedStatus === "approved" &&
        currentStatus !== "pending" &&
        currentStatus !== "rejected"
      ) {
        res.status(409).json({
          success: false,
          message: "Only a pending or rejected appointment can be approved",
        });

        return;
      }

      if (
        requestedStatus === "rejected" &&
        currentStatus !== "pending" &&
        currentStatus !== "approved"
      ) {
        res.status(409).json({
          success: false,
          message: "Only a pending or approved appointment can be rejected",
        });

        return;
      }

      if (requestedStatus === "approved" && currentStatus === "rejected") {
        const anotherActiveAppointment = await appointmentsCollection.findOne({
          _id: {
            $ne: appointment._id,
          },
          doctorId: getDoctorString(appointment.doctorId),
          patientUserId: getDoctorString(appointment.patientUserId),
          status: {
            $in: ACTIVE_APPOINTMENT_STATUSES,
          },
        });

        if (anotherActiveAppointment) {
          res.status(409).json({
            success: false,
            message:
              "This patient already has another pending or approved appointment with you.",
          });

          return;
        }
      }

      const now = new Date();

      const statusFields: Record<string, unknown> = {
        status: requestedStatus as AppointmentStatus,
        rejectionReason:
          requestedStatus === "rejected" ? rejectionReason : null,
        updatedAt: now,
      };

      if (requestedStatus === "approved") {
        statusFields.approvedAt = now;
        statusFields.rejectedAt = null;
        statusFields.rejectionReason = null;
      }

      if (requestedStatus === "completed") {
        statusFields.completedAt = now;
      }

      if (requestedStatus === "rejected") {
        statusFields.rejectedAt = now;
      }

      const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
        {
          _id: appointment._id,
        },
        {
          $set: statusFields,
        },
        {
          returnDocument: "after",
        },
      );

      res.status(200).json({
        success: true,
        message:
          requestedStatus === "approved"
            ? "Appointment approved successfully"
            : requestedStatus === "completed"
              ? "Consultation completed successfully."
              : "Appointment rejected successfully",
        appointment: updatedAppointment
          ? formatAppointment(updatedAppointment)
          : null,
      });
    } catch (error) {
      console.error("Update appointment status error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to update appointment status",
      });
    }
  },
);

/* =========================================================
   Admin Dashboard Statistics
========================================================= */

app.get(
  "/api/v1/admin/dashboard/stats",
  verifyToken,
  verifyAdmin,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    console.log("✅ Admin dashboard stats route called!"); // Debug log
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      // Get total patients
      const totalPatients = await database
        .collection("user")
        .countDocuments({ role: "patient" });

      // Get active patients
      const activePatients = await database
        .collection("user")
        .countDocuments({ role: "patient", status: "active" });

      // Get blocked patients
      const blockedPatients = await database
        .collection("user")
        .countDocuments({ role: "patient", status: "blocked" });

      // Get total doctors
      const totalDoctors = await database
        .collection("doctors")
        .countDocuments();

      // Get active doctors
      const activeDoctors = await database
        .collection("doctors")
        .countDocuments({ status: "active" });

      // Get blocked doctors
      const blockedDoctors = await database
        .collection("doctors")
        .countDocuments({ status: "blocked" });

      // Get appointment counts by status
      const appointmentCounts = await database
        .collection("appointments")
        .aggregate([
          {
            $group: {
              _id: "$status",
              count: { $sum: 1 },
            },
          },
        ])
        .toArray();

      // Create status count object with default values
      const statusCounts: Record<string, number> = {
        pending: 0,
        approved: 0,
        completed: 0,
        rejected: 0,
      };

      appointmentCounts.forEach((item) => {
        const status = item._id || "pending";
        if (status in statusCounts) {
          statusCounts[status] = item.count;
        }
      });

      // Get total appointments
      const totalAppointments = await database
        .collection("appointments")
        .countDocuments();

      // Get completed consultations
      const completedConsultations = statusCounts.completed;

      // Get monthly appointment trends (last 6 months)
      const sixMonthsAgo = new Date();
      sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

      const monthlyTrends = await database
        .collection("appointments")
        .aggregate([
          {
            $match: {
              createdAt: { $gte: sixMonthsAgo },
            },
          },
          {
            $group: {
              _id: {
                year: { $year: "$createdAt" },
                month: { $month: "$createdAt" },
              },
              count: { $sum: 1 },
              pending: {
                $sum: {
                  $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
                },
              },
              approved: {
                $sum: {
                  $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
                },
              },
              completed: {
                $sum: {
                  $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
                },
              },
              rejected: {
                $sum: {
                  $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
                },
              },
            },
          },
          {
            $sort: { "_id.year": 1, "_id.month": 1 },
          },
        ])
        .toArray();

      // Get appointment status breakdown for charts
      const statusColors: Record<string, string> = {
        pending: "#FBBF24",
        approved: "#60A5FA",
        completed: "#34D399",
        rejected: "#F87171",
      };

      const statusData = appointmentCounts.map((item) => ({
        name: item._id || "unknown",
        value: item.count,
        fill: statusColors[item._id] || "#9CA3AF",
      }));

      // Format monthly data for charts
      const monthNames = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
      ];
      const monthlyData = monthlyTrends.map((item) => ({
        month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
        pending: item.pending || 0,
        approved: item.approved || 0,
        completed: item.completed || 0,
        rejected: item.rejected || 0,
        total: item.count || 0,
      }));

      // Get recent appointments (last 10)
      const recentAppointments = await database
        .collection("appointments")
        .find()
        .sort({ createdAt: -1 })
        .limit(10)
        .toArray();

      res.status(200).json({
        success: true,
        data: {
          overview: {
            totalPatients,
            activePatients,
            blockedPatients,
            totalDoctors,
            activeDoctors,
            blockedDoctors,
            totalAppointments,
            completedConsultations,
            appointmentStatus: statusCounts,
          },
          charts: {
            appointmentStatus: statusData,
            monthlyTrends: monthlyData,
          },
          recentAppointments: recentAppointments.map((app) => ({
            id: app._id,
            patientName: app.patientName,
            doctorName: app.doctorName,
            specialization: app.specialization,
            appointmentDate: app.appointmentDate,
            appointmentTime: app.appointmentTime,
            status: app.status,
            createdAt: app.createdAt,
          })),
        },
      });
    } catch (error) {
      console.error("Dashboard stats error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to fetch dashboard statistics",
      });
    }
  },
);

/* =========================================================
   SebaSathi AI Health Assistant (Groq)
========================================================= */

type AIHealthMessageRole = "user" | "assistant";

type AIHealthUrgency = "routine" | "soon" | "urgent" | "emergency";

type AIHealthStreamStage =
  | "thinking"
  | "tool"
  | "answering"
  | "structuring"
  | "saving";

interface AIHealthMessage {
  role: AIHealthMessageRole;
  content: string;
}

interface AIHealthNavigationRoute {
  label: string;
  href: string;
  description: string;
}

interface AIHealthNavigationAction {
  label: string;
  href: string;
  reason: string;
}

interface AIHealthAssistantResponse {
  reply: string;
  urgencyLevel: AIHealthUrgency;
  suggestedSpecialists: string[];
  recommendedActions: string[];
  warningSigns: string[];
  followUpQuestions: string[];
  suggestedPrompts: string[];
  navigationActions: AIHealthNavigationAction[];
  decisionBasis: string;
  toolsUsed: string[];
  contextMemoryUsed: boolean;
  disclaimer: string;
}

interface AIHealthStoredMessage extends AIHealthMessage {
  id: string;
  assistant?: AIHealthAssistantResponse;
  createdAt: Date;
}

interface AIHealthSummaryReport {
  reportTitle: string;
  conciseSummary: string;
  chiefConcerns: string[];
  symptoms: string[];
  durationAndPattern: string;
  severity: string;
  urgencyLevel: AIHealthUrgency;
  redFlags: string[];
  suggestedSpecialists: string[];
  selfCareGuidance: string[];
  questionsForDoctor: string[];
  emergencyAdvice: string;
  disclaimer: string;
}

interface AIHealthConversationDocument {
  _id?: ObjectId;
  title?: string;
  messages: AIHealthStoredMessage[];
  summaryHistoryId?: string | null;
  summaryReport?: AIHealthSummaryReport | null;
  updatedAt?: Date;
  lastMessageAt?: Date;
}

interface AIHealthApplicationContext {
  user: {
    id: string;
    name: string;
    role: UserRole;
  };
  routes: AIHealthNavigationRoute[];
  doctorDirectory: {
    activeDoctorCount: number;
    specializations: string[];
    highlightedDoctors: Array<{
      id: string;
      name: string;
      specialization: string;
      hospital: string;
      ratingAverage: number;
    }>;
  } | null;
  appointmentContext: {
    total: number;
    counts: Record<string, number>;
    recentAppointments: Array<{
      id: string;
      doctorName: string;
      patientName: string;
      specialization: string;
      appointmentDate: string;
      appointmentTime: string;
      status: string;
    }>;
  } | null;
  recentHealthHistory: Array<{
    id: string;
    title: string;
    urgencyLevel: AIHealthUrgency;
    updatedAt: string | null;
  }>;
  toolsUsed: string[];
  contextMemoryUsed: boolean;
}

const aiHealthRateLimit = new Map<
  string,
  {
    startedAt: number;
    count: number;
  }
>();

const verifyAIHealthRateLimit = (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction,
): void => {
  const key = req.userId || req.ip || "anonymous";
  const now = Date.now();
  const windowLength = 10 * 60 * 1000;
  const maximumRequests = 40;
  const current = aiHealthRateLimit.get(key);

  if (!current || now - current.startedAt >= windowLength) {
    aiHealthRateLimit.set(key, {
      startedAt: now,
      count: 1,
    });

    next();
    return;
  }

  if (current.count >= maximumRequests) {
    res.status(429).json({
      success: false,
      message:
        "You have sent too many AI requests. Please try again after a few minutes.",
      code: "AI_RATE_LIMITED",
    });

    return;
  }

  current.count += 1;
  aiHealthRateLimit.set(key, current);
  next();
};

const createAIHealthMessageId = (): string => {
  return new ObjectId().toHexString();
};

const getAIHealthArray = (value: unknown, maximumItems = 8): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item) => getDoctorString(item))
    .filter(Boolean)
    .slice(0, maximumItems);
};

const getAIHealthUrgency = (value: unknown): AIHealthUrgency => {
  return value === "soon" || value === "urgent" || value === "emergency"
    ? value
    : "routine";
};

const getAIHealthBoolean = (value: unknown): boolean => value === true;

const extractAIHealthJson = (content: string): Record<string, unknown> => {
  const trimmed = content.trim();
  const withoutFence = trimmed
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/i, "")
    .trim();

  try {
    const parsed = JSON.parse(withoutFence) as unknown;

    if (typeof parsed === "object" && parsed !== null) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    const firstBrace = withoutFence.indexOf("{");
    const lastBrace = withoutFence.lastIndexOf("}");

    if (firstBrace >= 0 && lastBrace > firstBrace) {
      const parsed = JSON.parse(
        withoutFence.slice(firstBrace, lastBrace + 1),
      ) as unknown;

      if (typeof parsed === "object" && parsed !== null) {
        return parsed as Record<string, unknown>;
      }
    }
  }

  throw new Error("Groq returned an invalid structured response");
};

const normalizeAIHealthMessages = (
  value: unknown,
  options: {
    requireLatestUser: boolean;
    maximumMessages?: number;
    maximumCharacters?: number;
  },
):
  | {
      success: true;
      messages: AIHealthMessage[];
    }
  | {
      success: false;
      message: string;
    } => {
  if (!Array.isArray(value)) {
    return {
      success: false,
      message: "A conversation message list is required",
    };
  }

  const maximumMessages = options.maximumMessages ?? 30;
  const maximumCharacters = options.maximumCharacters ?? 30000;
  const messages: AIHealthMessage[] = [];
  let totalCharacters = 0;

  for (const rawMessage of value.slice(-maximumMessages)) {
    if (typeof rawMessage !== "object" || rawMessage === null) {
      continue;
    }

    const message = rawMessage as Record<string, unknown>;
    const role = message.role;
    const content = getDoctorString(message.content);

    if ((role !== "user" && role !== "assistant") || !content) {
      continue;
    }

    if (content.length > 4000) {
      return {
        success: false,
        message: "Each chat message cannot contain more than 4000 characters",
      };
    }

    totalCharacters += content.length;

    if (totalCharacters > maximumCharacters) {
      return {
        success: false,
        message:
          "This conversation is too long. Please generate a summary and start a new conversation.",
      };
    }

    messages.push({
      role,
      content,
    });
  }

  if (messages.length === 0) {
    return {
      success: false,
      message: "At least one valid chat message is required",
    };
  }

  if (!messages.some((message) => message.role === "user")) {
    return {
      success: false,
      message: "At least one user message is required",
    };
  }

  if (
    options.requireLatestUser &&
    messages[messages.length - 1]?.role !== "user"
  ) {
    return {
      success: false,
      message: "The latest conversation message must be from the user",
    };
  }

  return {
    success: true,
    messages,
  };
};

const PUBLIC_AI_HEALTH_NAVIGATION_ROUTES: AIHealthNavigationRoute[] = [
  {
    label: "Home",
    href: "/",
    description: "Open the SebaSathi home page.",
  },
  {
    label: "Find Doctors",
    href: "/find-doctors",
    description: "Find active doctors and filter by specialization.",
  },
  {
    label: "AI Health Assistant",
    href: "/ai-health-assistant",
    description: "Continue using the SebaSathi AI Health Assistant.",
  },
  {
    label: "About Us",
    href: "/about",
    description: "Learn more about SebaSathi and its healthcare services.",
  },
  {
    label: "Contact",
    href: "/contact",
    description: "Open the SebaSathi contact page.",
  },
];

const ROLE_AI_HEALTH_NAVIGATION_ROUTES: Record<
  UserRole,
  AIHealthNavigationRoute[]
> = {
  patient: [
    {
      label: "Patient Overview",
      href: "/dashboard/patient",
      description: "Open the patient's dashboard overview.",
    },
    {
      label: "My Appointments",
      href: "/dashboard/patient/appointments",
      description: "View the patient's appointment requests and statuses.",
    },
    {
      label: "Prescriptions",
      href: "/dashboard/patient/prescriptions",
      description: "View the patient's saved prescriptions.",
    },
    {
      label: "Consultations",
      href: "/dashboard/patient/consultations",
      description: "View the patient's consultation records.",
    },
    {
      label: "AI Health History",
      href: "/dashboard/patient/ai-health-history",
      description: "Review saved AI-generated health summaries.",
    },
    {
      label: "My Profile",
      href: "/dashboard/patient/my-profile",
      description: "Open the patient's profile settings.",
    },
  ],
  doctor: [
    {
      label: "Doctor Overview",
      href: "/dashboard/doctor",
      description: "Open the doctor's dashboard overview.",
    },
    {
      label: "Appointments",
      href: "/dashboard/doctor/patients-appointments",
      description: "View appointments assigned to the signed-in doctor.",
    },
    {
      label: "My Patients",
      href: "/dashboard/doctor/patients",
      description: "View the doctor's patient list.",
    },
    {
      label: "Prescriptions",
      href: "/dashboard/doctor/prescriptions",
      description: "Create or review doctor prescription records.",
    },
    {
      label: "Consultation Records",
      href: "/dashboard/doctor/consultations",
      description: "View the doctor's consultation records.",
    },
    {
      label: "Availability",
      href: "/dashboard/doctor/availability",
      description: "Manage the doctor's availability schedule.",
    },
    {
      label: "My Profile",
      href: "/dashboard/doctor/my-profile",
      description: "Open the doctor's profile settings.",
    },
  ],
  admin: [
    {
      label: "Admin Overview",
      href: "/dashboard/admin",
      description: "Open the administrator dashboard overview.",
    },
    {
      label: "Manage Users",
      href: "/dashboard/admin/users",
      description: "Open administrator user management.",
    },
    {
      label: "Manage Doctors",
      href: "/dashboard/admin/doctors",
      description: "Open administrator doctor management.",
    },
    {
      label: "Manage Appointments",
      href: "/dashboard/admin/appointments",
      description: "Open administrator appointment management.",
    },
    {
      label: "My Profile",
      href: "/dashboard/admin/my-profile",
      description: "Open the administrator's profile settings.",
    },
  ],
};

const AI_HEALTH_NAVIGATION_ROUTE_ALIASES: Record<string, string> = {
  "/doctors": "/find-doctors",
  "/dashboard/doctor/appointments": "/dashboard/doctor/patients-appointments",
};

const normalizeAIHealthNavigationHref = (href: string): string => {
  return AI_HEALTH_NAVIGATION_ROUTE_ALIASES[href] || href;
};

const getAllAIHealthNavigationRoutes = (): AIHealthNavigationRoute[] => [
  ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
  ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.patient,
  ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.doctor,
  ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.admin,
];

const getAIHealthNavigationRoutes = (
  role: UserRole,
): AIHealthNavigationRoute[] => [
  ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
  ...ROLE_AI_HEALTH_NAVIGATION_ROUTES[role],
];

const getAIHealthNavigationActions = (
  value: unknown,
  allowedRoutes: AIHealthNavigationRoute[],
): AIHealthNavigationAction[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const allowedByHref = new Map(
    allowedRoutes.map((route) => [route.href, route] as const),
  );

  const actions: AIHealthNavigationAction[] = [];

  for (const rawAction of value) {
    if (typeof rawAction !== "object" || rawAction === null) {
      continue;
    }

    const action = rawAction as Record<string, unknown>;
    const href = normalizeAIHealthNavigationHref(getDoctorString(action.href));
    const allowedRoute = allowedByHref.get(href);

    if (!allowedRoute) {
      continue;
    }

    actions.push({
      label: getDoctorString(action.label) || allowedRoute.label,
      href,
      reason: getDoctorString(action.reason) || allowedRoute.description,
    });

    if (actions.length >= 3) {
      break;
    }
  }

  return actions;
};

const formatAIHealthAssistantResponse = (
  data: Record<string, unknown>,
  emergencyDetected: boolean,
  context?: AIHealthApplicationContext,
): AIHealthAssistantResponse => {
  const urgencyLevel = emergencyDetected
    ? "emergency"
    : getAIHealthUrgency(data.urgencyLevel);

  const reply =
    getDoctorString(data.reply) ||
    "Please describe the symptoms, duration and severity a little more clearly.";

  const followUpQuestions = getAIHealthArray(data.followUpQuestions, 3);
  const suggestedPrompts = getAIHealthArray(data.suggestedPrompts, 4);
  const allowedRoutes = context?.routes || getAllAIHealthNavigationRoutes();
  const toolsUsed = context?.toolsUsed.length
    ? context.toolsUsed
    : getAIHealthArray(data.toolsUsed, 8);
  const contextMemoryUsed = context
    ? context.contextMemoryUsed
    : getAIHealthBoolean(data.contextMemoryUsed);

  return {
    reply,
    urgencyLevel,
    suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 3),
    recommendedActions: getAIHealthArray(data.recommendedActions, 5),
    warningSigns: emergencyDetected
      ? Array.from(
          new Set([
            "Your description may include an emergency warning sign.",
            ...getAIHealthArray(data.warningSigns, 4),
          ]),
        ).slice(0, 4)
      : getAIHealthArray(data.warningSigns, 4),
    followUpQuestions,
    suggestedPrompts:
      suggestedPrompts.length > 0
        ? suggestedPrompts
        : followUpQuestions.slice(0, 3),
    navigationActions: getAIHealthNavigationActions(
      data.navigationActions,
      allowedRoutes,
    ),
    decisionBasis:
      getDoctorString(data.decisionBasis) ||
      "This guidance is based on the symptoms, duration, severity, warning signs and relevant SebaSathi application context available in this conversation.",
    toolsUsed,
    contextMemoryUsed,
    disclaimer:
      getDoctorString(data.disclaimer) ||
      "General guidance only; this is not a diagnosis or prescription.",
  };
};

const getStoredAIHealthMessages = (value: unknown): AIHealthStoredMessage[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((rawMessage): AIHealthStoredMessage | null => {
      if (typeof rawMessage !== "object" || rawMessage === null) {
        return null;
      }

      const message = rawMessage as Record<string, unknown>;
      const role = message.role;
      const content = getDoctorString(message.content);

      if ((role !== "user" && role !== "assistant") || !content) {
        return null;
      }

      const assistant =
        typeof message.assistant === "object" && message.assistant !== null
          ? formatAIHealthAssistantResponse(
              message.assistant as Record<string, unknown>,
              false,
            )
          : undefined;

      const createdAtValue = message.createdAt;
      const createdAt =
        createdAtValue instanceof Date
          ? createdAtValue
          : new Date(
              typeof createdAtValue === "string" ||
                typeof createdAtValue === "number"
                ? createdAtValue
                : Date.now(),
            );

      return {
        id: getDoctorString(message.id) || createAIHealthMessageId(),
        role,
        content,
        ...(assistant ? { assistant } : {}),
        createdAt: Number.isNaN(createdAt.getTime()) ? new Date() : createdAt,
      };
    })
    .filter((message): message is AIHealthStoredMessage => message !== null);
};

const hasEmergencyWarning = (messages: AIHealthMessage[]): boolean => {
  const text = messages
    .filter((message) => message.role === "user")
    .map((message) => message.content)
    .join(" ")
    .toLowerCase();

  const emergencyPatterns = [
    /severe chest pain/,
    /cannot breathe/,
    /can't breathe/,
    /difficulty breathing/,
    /heavy bleeding/,
    /unconscious/,
    /not responding/,
    /seizure/,
    /stroke symptoms/,
    /face droop/,
    /suicid(?:e|al)/,
    /kill myself/,
    /বুকে তীব্র ব্যথা/,
    /শ্বাস নিতে পারছি না/,
    /শ্বাসকষ্ট/,
    /অতিরিক্ত রক্তপাত/,
    /অজ্ঞান/,
    /খিঁচুনি/,
    /আত্মহত্যা/,
  ];

  return emergencyPatterns.some((pattern) => pattern.test(text));
};

const callGroqAI = async (
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string;
  }>,
  temperature: number,
  maximumOutputTokens: number,
): Promise<Record<string, unknown>> => {
  if (!groqApiKey) {
    throw new Error("GROQ_API_KEY is missing from the backend .env file");
  }

  const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${groqApiKey}`,
      "content-type": "application/json",
      accept: "application/json",
    },
    body: JSON.stringify({
      model: groqModel,
      messages,
      temperature,
      max_completion_tokens: maximumOutputTokens,
      response_format: {
        type: "json_object",
      },
    }),
  });

  const responseData = (await response.json().catch(() => null)) as Record<
    string,
    unknown
  > | null;

  if (!response.ok) {
    const errorObject =
      typeof responseData?.error === "object" && responseData.error !== null
        ? (responseData.error as Record<string, unknown>)
        : null;

    const providerMessage = getDoctorString(errorObject?.message);

    throw new Error(
      providerMessage || `Groq request failed with status ${response.status}`,
    );
  }

  const choices = Array.isArray(responseData?.choices)
    ? responseData.choices
    : [];

  const firstChoice = choices[0];

  if (typeof firstChoice !== "object" || firstChoice === null) {
    throw new Error("Groq did not return an assistant response");
  }

  const choice = firstChoice as Record<string, unknown>;
  const message =
    typeof choice.message === "object" && choice.message !== null
      ? (choice.message as Record<string, unknown>)
      : null;

  const content = getDoctorString(message?.content);

  if (!content) {
    throw new Error("Groq returned an empty assistant response");
  }

  return extractAIHealthJson(content);
};

const callGroqTextStream = async (
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string;
  }>,
  onDelta: (delta: string) => void,
  signal?: AbortSignal,
): Promise<string> => {
  if (!groqApiKey) {
    throw new Error("GROQ_API_KEY is missing from the backend .env file");
  }

  const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${groqApiKey}`,
      "content-type": "application/json",
      accept: "text/event-stream",
    },
    body: JSON.stringify({
      model: groqModel,
      messages,
      temperature: 0.25,
      max_completion_tokens: 1100,
      stream: true,
    }),
    signal,
  });

  if (!response.ok) {
    const responseData = (await response.json().catch(() => null)) as Record<
      string,
      unknown
    > | null;
    const errorObject =
      typeof responseData?.error === "object" && responseData.error !== null
        ? (responseData.error as Record<string, unknown>)
        : null;
    const providerMessage = getDoctorString(errorObject?.message);

    throw new Error(
      providerMessage || `Groq request failed with status ${response.status}`,
    );
  }

  if (!response.body) {
    throw new Error("Groq streaming response body is unavailable");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completeText = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();

      if (!trimmed.startsWith("data:")) {
        continue;
      }

      const payload = trimmed.slice(5).trim();

      if (!payload || payload === "[DONE]") {
        continue;
      }

      const parsed = JSON.parse(payload) as Record<string, unknown>;
      const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
      const firstChoice = choices[0];

      if (typeof firstChoice !== "object" || firstChoice === null) {
        continue;
      }

      const delta = (firstChoice as Record<string, unknown>).delta;

      if (typeof delta !== "object" || delta === null) {
        continue;
      }

      const rawContent = (delta as Record<string, unknown>).content;
      const content = typeof rawContent === "string" ? rawContent : "";

      if (!content) {
        continue;
      }

      completeText += content;
      onDelta(content);
    }
  }

  const finalText = completeText.trim();

  if (!finalText) {
    throw new Error("Groq returned an empty streamed response");
  }

  return finalText;
};

const formatAIHealthSummary = (
  data: Record<string, unknown>,
): AIHealthSummaryReport => {
  return {
    reportTitle:
      getDoctorString(data.reportTitle) || "AI Health Conversation Summary",
    conciseSummary:
      getDoctorString(data.conciseSummary) ||
      "A concise summary could not be generated.",
    chiefConcerns: getAIHealthArray(data.chiefConcerns, 6),
    symptoms: getAIHealthArray(data.symptoms, 10),
    durationAndPattern:
      getDoctorString(data.durationAndPattern) || "Not clearly stated",
    severity: getDoctorString(data.severity) || "Not clearly stated",
    urgencyLevel: getAIHealthUrgency(data.urgencyLevel),
    redFlags: getAIHealthArray(data.redFlags, 6),
    suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 5),
    selfCareGuidance: getAIHealthArray(data.selfCareGuidance, 6),
    questionsForDoctor: getAIHealthArray(data.questionsForDoctor, 6),
    emergencyAdvice:
      getDoctorString(data.emergencyAdvice) ||
      "Seek urgent in-person medical care if symptoms become severe or new warning signs appear.",
    disclaimer:
      getDoctorString(data.disclaimer) ||
      "This AI-generated summary is not a diagnosis or prescription.",
  };
};

const createAIHealthConversationTitle = (message: string): string => {
  const normalized = message.replace(/\s+/g, " ").trim();
  const words = normalized.split(" ").filter(Boolean).slice(0, 7);
  const title = words.join(" ");

  if (!title) {
    return "New health chat";
  }

  return normalized.length > title.length ? `${title}…` : title;
};

const getAIHealthOwnerFilter = (userId: string): Filter<Document> => {
  return {
    $or: [
      {
        userId,
      },
      {
        patientUserId: userId,
      },
    ],
  };
};

const formatAIHealthConversationMessage = (message: AIHealthStoredMessage) => {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    assistant: message.assistant || null,
    createdAt: formatDoctorDate(message.createdAt),
  };
};

const formatAIHealthConversation = (conversation: Document) => {
  const userId =
    getDoctorString(conversation.userId) ||
    getDoctorString(conversation.patientUserId);

  const userRole: UserRole =
    conversation.userRole === "admin" ||
    conversation.userRole === "doctor" ||
    conversation.userRole === "patient"
      ? conversation.userRole
      : "patient";

  const messages = getStoredAIHealthMessages(conversation.messages);

  return {
    id: getDoctorDocumentId(conversation),
    title: getDoctorString(conversation.title) || "New health chat",
    userId,
    userRole,
    userName:
      getDoctorString(conversation.userName) ||
      getDoctorString(conversation.patientName),
    userEmail:
      normalizeDoctorEmail(conversation.userEmail) ||
      normalizeDoctorEmail(conversation.patientEmail),
    userImage:
      getDoctorString(conversation.userImage) ||
      getDoctorString(conversation.patientImage) ||
      null,
    messages: messages.map(formatAIHealthConversationMessage),
    messageCount: messages.length,
    summaryHistoryId: getDoctorString(conversation.summaryHistoryId) || null,
    summaryReport:
      typeof conversation.summaryReport === "object" &&
      conversation.summaryReport !== null
        ? conversation.summaryReport
        : null,
    createdAt: formatDoctorDate(conversation.createdAt),
    updatedAt: formatDoctorDate(conversation.updatedAt),
    lastMessageAt: formatDoctorDate(
      conversation.lastMessageAt || conversation.updatedAt,
    ),
  };
};

const formatAIHealthHistory = (history: Document) => {
  const userId =
    getDoctorString(history.userId) || getDoctorString(history.patientUserId);

  const userName =
    getDoctorString(history.userName) || getDoctorString(history.patientName);

  const userEmail =
    normalizeDoctorEmail(history.userEmail) ||
    normalizeDoctorEmail(history.patientEmail);

  const userRole: UserRole =
    history.userRole === "admin" ||
    history.userRole === "doctor" ||
    history.userRole === "patient"
      ? history.userRole
      : "patient";

  return {
    id: getDoctorDocumentId(history),
    conversationId: getDoctorString(history.conversationId) || null,
    conversationTitle: getDoctorString(history.conversationTitle) || null,
    userId,
    userRole,
    userName,
    userEmail,
    userImage:
      getDoctorString(history.userImage) ||
      getDoctorString(history.patientImage) ||
      null,
    patientUserId: userId,
    patientName: userName,
    patientEmail: userEmail,
    provider: getDoctorString(history.provider),
    model: getDoctorString(history.model),
    report:
      typeof history.report === "object" && history.report !== null
        ? history.report
        : null,
    messages: Array.isArray(history.messages) ? history.messages : [],
    createdAt: formatDoctorDate(history.createdAt),
    updatedAt: formatDoctorDate(history.updatedAt),
  };
};

const getAIHealthConversationForUser = async (
  userId: string,
  conversationId: string,
): Promise<Document | null> => {
  if (!database || !conversationId) {
    return null;
  }

  return database.collection(AI_HEALTH_CHAT_COLLECTION).findOne({
    $and: [getDoctorFilter(conversationId), getAIHealthOwnerFilter(userId)],
  });
};

const detectAIHealthApplicationIntents = (message: string) => {
  const normalized = message.toLowerCase();

  return {
    appointment:
      /appointment|booking|schedule|pending|approved|rejected|অ্যাপয়েন্টমেন্ট|অ্যাপয়েন্টমেন্ট|বুকিং|সিডিউল|পেন্ডিং|এপ্রুভ/.test(
        normalized,
      ),
    history:
      /history|summary|report|previous chat|old chat|হিস্ট্রি|সামারি|রিপোর্ট|পুরোনো চ্যাট/.test(
        normalized,
      ),
    navigation:
      /open|go to|take me|navigate|where is|show page|dashboard|খুলে দাও|নিয়ে যাও|নিয়ে যাও|কোথায়|কোথায়|ড্যাশবোর্ড/.test(
        normalized,
      ),
    doctor:
      /doctor|specialist|specialization|cardio|derma|neuro|medicine|surgeon|ডাক্তার|বিশেষজ্ঞ|স্পেশালিস্ট|কার্ডিও|ডার্মা|নিউরো/.test(
        normalized,
      ),
  };
};

const buildAIHealthApplicationContext = async (
  currentUser: Document,
  latestMessage: string,
  existingMessages: AIHealthStoredMessage[],
): Promise<AIHealthApplicationContext> => {
  if (!database) {
    throw new Error("Database is not connected");
  }

  const userId = getDoctorDocumentId(currentUser);
  const role = getNormalizedUserRole(currentUser);
  const userName = getDoctorString(currentUser.name) || "User";
  const intents = detectAIHealthApplicationIntents(latestMessage);
  const routes = getAIHealthNavigationRoutes(role);
  const toolsUsed = ["SebaSathi navigation map", "SebaSathi doctor directory"];
  const contextMemoryUsed = existingMessages.length > 0;

  if (contextMemoryUsed) {
    toolsUsed.push("Conversation memory");
  }

  const [doctorDocuments, specializations, activeDoctorCount] =
    await Promise.all([
      database
        .collection("doctors")
        .find(
          { status: "active" },
          {
            projection: {
              name: 1,
              specialization: 1,
              hospital: 1,
              chamber: 1,
              ratingAverage: 1,
            },
          },
        )
        .sort({ ratingAverage: -1, ratingCount: -1, createdAt: -1 })
        .limit(8)
        .toArray(),
      database.collection("doctors").distinct("specialization", {
        status: "active",
      }),
      database.collection("doctors").countDocuments({ status: "active" }),
    ]);

  const doctorDirectory = {
    activeDoctorCount,
    specializations: specializations
      .map((value) => getDoctorString(value))
      .filter(Boolean)
      .sort((a, b) => a.localeCompare(b))
      .slice(0, 30),
    highlightedDoctors: doctorDocuments.map((doctor) => ({
      id: getDoctorDocumentId(doctor),
      name: getDoctorString(doctor.name),
      specialization: getDoctorString(doctor.specialization),
      hospital:
        getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
      ratingAverage: Number.isFinite(Number(doctor.ratingAverage))
        ? Number(Number(doctor.ratingAverage).toFixed(1))
        : 0,
    })),
  };

  let appointmentContext: AIHealthApplicationContext["appointmentContext"] =
    null;

  if (intents.appointment || intents.navigation) {
    toolsUsed.push("Appointment lookup");

    const appointmentFilter: Filter<Document> =
      role === "patient"
        ? { patientUserId: userId }
        : role === "doctor"
          ? { doctorUserId: userId }
          : {};

    const [statusCounts, recentAppointments, total] = await Promise.all([
      database
        .collection("appointments")
        .aggregate([
          { $match: appointmentFilter },
          {
            $group: {
              _id: "$status",
              count: { $sum: 1 },
            },
          },
        ])
        .toArray(),
      database
        .collection("appointments")
        .find(appointmentFilter)
        .sort({ updatedAt: -1, createdAt: -1 })
        .limit(5)
        .toArray(),
      database.collection("appointments").countDocuments(appointmentFilter),
    ]);

    appointmentContext = {
      total,
      counts: Object.fromEntries(
        statusCounts.map((item) => [
          getDoctorString(item._id) || "unknown",
          Number(item.count) || 0,
        ]),
      ),
      recentAppointments: recentAppointments.map((appointment) => ({
        id: getDoctorDocumentId(appointment),
        doctorName: getDoctorString(appointment.doctorName),
        patientName: getDoctorString(appointment.patientName),
        specialization: getDoctorString(appointment.specialization),
        appointmentDate: getDoctorString(appointment.appointmentDate),
        appointmentTime: getDoctorString(appointment.appointmentTime),
        status: getDoctorString(appointment.status) || "pending",
      })),
    };
  }

  let recentHealthHistory: AIHealthApplicationContext["recentHealthHistory"] =
    [];

  if (intents.history || intents.navigation) {
    toolsUsed.push("Saved AI health history lookup");

    const historyDocuments = await database
      .collection(AI_HEALTH_HISTORY_COLLECTION)
      .find(getAIHealthOwnerFilter(userId))
      .sort({ updatedAt: -1, createdAt: -1 })
      .limit(5)
      .toArray();

    recentHealthHistory = historyDocuments.map((history) => {
      const report =
        typeof history.report === "object" && history.report !== null
          ? (history.report as Record<string, unknown>)
          : {};

      return {
        id: getDoctorDocumentId(history),
        title:
          getDoctorString(history.conversationTitle) ||
          getDoctorString(report.reportTitle) ||
          "AI Health Summary",
        urgencyLevel: getAIHealthUrgency(report.urgencyLevel),
        updatedAt: formatDoctorDate(history.updatedAt || history.createdAt),
      };
    });
  }

  if (intents.doctor) {
    toolsUsed.push("Specialist matching context");
  }

  return {
    user: {
      id: userId,
      name: userName,
      role,
    },
    routes,
    doctorDirectory,
    appointmentContext,
    recentHealthHistory,
    toolsUsed: Array.from(new Set(toolsUsed)),
    contextMemoryUsed,
  };
};

const buildAIHealthNaturalResponsePrompt = (
  context: AIHealthApplicationContext,
): string => `You are SebaSathi AI Health Assistant, an advanced conversational assistant integrated into a Bangladesh-oriented healthcare application.

You must do more than simple text generation. Use conversation memory and the supplied SebaSathi application context to answer questions, reason about next steps, help the user navigate the application, and ask useful follow-up questions when information is missing.

Signed-in user:
${JSON.stringify(context.user)}

SebaSathi application context retrieved by backend tools:
${JSON.stringify({
  routes: context.routes,
  doctorDirectory: context.doctorDirectory,
  appointmentContext: context.appointmentContext,
  recentHealthHistory: context.recentHealthHistory,
  toolsUsed: context.toolsUsed,
})}

Behavior requirements:
- Answer health questions and SebaSathi application questions naturally.
- Use previous conversation messages to understand references such as “it”, “that problem”, “same pain”, or “what should I do next”.
- When application data is available, use it accurately. Never invent appointments, doctors, history, counts, status, dates or routes.
- If the user asks where to go in the application, explain the correct page and mention the relevant route label naturally.
- Explain the practical basis for recommendations without revealing hidden chain-of-thought.
- Ask concise follow-up questions when key details are missing.
- Match the user's language: easy Bangla, Banglish or English.
- For health guidance, never confirm a diagnosis, prescribe medicine, provide individualized doses, or advise stopping prescribed treatment.
- Emergency warning signs require immediate emergency-care advice.
- Usually write 5-9 clear sentences and approximately 120-240 words when enough information exists.
- Return only the natural conversational answer. Do not return JSON, markdown tables, internal IDs or hidden reasoning.`;

const buildAIHealthMetadataPrompt = (
  context: AIHealthApplicationContext,
  latestUserMessage: string,
  assistantReply: string,
): string => `Create safe structured metadata for a completed SebaSathi AI assistant reply.

User message:
${latestUserMessage}

Assistant reply:
${assistantReply}

Allowed navigation routes:
${JSON.stringify(context.routes)}

Backend tools already used:
${JSON.stringify(context.toolsUsed)}

Return ONLY valid JSON with this exact shape:
{
  "urgencyLevel": "routine | soon | urgent | emergency",
  "suggestedSpecialists": ["maximum three specialist categories that exist in or reasonably map to the doctor directory"],
  "recommendedActions": ["maximum five safe practical actions"],
  "warningSigns": ["maximum four important warning signs"],
  "followUpQuestions": ["maximum three useful follow-up questions"],
  "suggestedPrompts": ["maximum four short prompts the user can click to continue the conversation"],
  "navigationActions": [
    {
      "label": "must correspond to an allowed route",
      "href": "must exactly match one allowed route href",
      "reason": "short explanation of why this page is relevant"
    }
  ],
  "decisionBasis": "one or two concise user-facing sentences explaining which reported facts or application context influenced the guidance, without exposing private chain-of-thought",
  "toolsUsed": ["copy only tools actually listed above"],
  "contextMemoryUsed": ${context.contextMemoryUsed ? "true" : "false"},
  "disclaimer": "one short medical disclaimer"
}

Do not invent app data. Include navigationActions only when useful. Suggested prompts must be directly usable as the user's next message.`;

const writeAIHealthStreamEvent = (
  res: Response,
  event: Record<string, unknown>,
): void => {
  if (!res.writableEnded) {
    res.write(`${JSON.stringify(event)}\n`);
  }
};

const startAIHealthStream = (res: Response): void => {
  res.status(200);
  res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();
};

const writeAIHealthStatus = (
  res: Response,
  stage: AIHealthStreamStage,
  message: string,
  toolsUsed: string[] = [],
): void => {
  writeAIHealthStreamEvent(res, {
    type: "status",
    stage,
    message,
    toolsUsed,
  });
};

/* =========================================================
   AI Health access status
========================================================= */

app.get(
  "/api/v1/ai-health/access",
  verifyToken,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          allowed: false,
          message: "User account was not found",
        });

        return;
      }

      const role = getNormalizedUserRole(currentUser);
      const status = getNormalizedUserStatus(currentUser);
      const allowed = status === "active";

      res.status(200).json({
        success: true,
        authenticated: true,
        allowed,
        role,
        status,
        user: {
          id: getDoctorDocumentId(currentUser),
          name: getDoctorString(currentUser.name),
          email: normalizeDoctorEmail(currentUser.email),
          image: getDoctorString(currentUser.image) || null,
        },
        message: allowed
          ? "Your active account can use SebaSathi AI Health Assistant."
          : "Your account is blocked. Contact the administrator to use the AI Health Assistant.",
      });
    } catch (error) {
      console.error("AI Health access error:", error);

      res.status(500).json({
        success: false,
        allowed: false,
        message: "Failed to verify AI Health access",
      });
    }
  },
);

/* =========================================================
   AI Health conversation history
========================================================= */

app.get(
  "/api/v1/ai-health/conversations",
  verifyToken,
  verifyAnyActiveUser,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const limit = getPositiveInteger(req.query.limit, 100, 100);
      const conversations = await database
        .collection(AI_HEALTH_CHAT_COLLECTION)
        .find(getAIHealthOwnerFilter(userId))
        .sort({
          lastMessageAt: -1,
          updatedAt: -1,
          _id: -1,
        })
        .limit(limit)
        .toArray();

      res.status(200).json({
        success: true,
        conversations: conversations.map(formatAIHealthConversation),
      });
    } catch (error) {
      console.error("Get AI conversations error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to retrieve AI conversations",
      });
    }
  },
);

app.post(
  "/api/v1/ai-health/conversations",
  verifyToken,
  verifyAnyActiveUser,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const now = new Date();
      const userId = getDoctorDocumentId(currentUser);
      const userRole = getNormalizedUserRole(currentUser);
      const userName = getDoctorString(currentUser.name);
      const userEmail = normalizeDoctorEmail(currentUser.email);
      const userImage = getDoctorString(currentUser.image) || null;
      const requestedTitle = getDoctorString(req.body.title).slice(0, 80);

      const conversationDocument = {
        title: requestedTitle || "New health chat",
        userId,
        userRole,
        userName,
        userEmail,
        userImage,
        patientUserId: userId,
        patientName: userName,
        patientEmail: userEmail,
        patientImage: userImage,
        messages: [] as AIHealthStoredMessage[],
        summaryHistoryId: null,
        summaryReport: null,
        createdAt: now,
        updatedAt: now,
        lastMessageAt: now,
      };

      const collection = database.collection(AI_HEALTH_CHAT_COLLECTION);
      const insertResult = await collection.insertOne(conversationDocument);
      const conversation = await collection.findOne({
        _id: insertResult.insertedId,
      });

      res.status(201).json({
        success: true,
        message: "New AI health conversation created",
        conversation: conversation
          ? formatAIHealthConversation(conversation)
          : {
              id: insertResult.insertedId.toHexString(),
              ...conversationDocument,
              messageCount: 0,
            },
      });
    } catch (error) {
      console.error("Create AI conversation error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to create AI conversation",
      });
    }
  },
);

app.get(
  "/api/v1/ai-health/conversations/:conversationId",
  verifyToken,
  verifyAnyActiveUser,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const conversation = await getAIHealthConversationForUser(
        getDoctorDocumentId(currentUser),
        getDoctorString(req.params.conversationId),
      );

      if (!conversation) {
        res.status(404).json({
          success: false,
          message: "AI health conversation was not found",
        });
        return;
      }

      res.status(200).json({
        success: true,
        conversation: formatAIHealthConversation(conversation),
      });
    } catch (error) {
      console.error("Get AI conversation details error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to retrieve AI conversation",
      });
    }
  },
);

app.delete(
  "/api/v1/ai-health/conversations/:conversationId",
  verifyToken,
  verifyAnyActiveUser,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const conversationId = getDoctorString(req.params.conversationId);
      const conversation = await getAIHealthConversationForUser(
        userId,
        conversationId,
      );

      if (!conversation) {
        res.status(404).json({
          success: false,
          message: "AI health conversation was not found",
        });
        return;
      }

      await database.collection(AI_HEALTH_CHAT_COLLECTION).deleteOne({
        _id: conversation._id,
      });

      res.status(200).json({
        success: true,
        message: "AI health conversation deleted successfully",
        deletedConversationId: conversationId,
      });
    } catch (error) {
      console.error("Delete AI conversation error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to delete AI conversation",
      });
    }
  },
);

/* =========================================================
   Advanced streamed AI Health message exchange
========================================================= */

app.post(
  "/api/v1/ai-health/conversations/:conversationId/messages/stream",
  verifyToken,
  verifyAnyActiveUser,
  verifyAIHealthRateLimit,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    let streamStarted = false;
    const abortController = new AbortController();

    res.on("close", () => {
      if (!res.writableEnded) {
        abortController.abort();
      }
    });

    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const conversationId = getDoctorString(req.params.conversationId);
      const content = getDoctorString(req.body.message);

      if (!content) {
        res.status(400).json({
          success: false,
          message: "A health or application question is required",
        });
        return;
      }

      if (content.length > 4000) {
        res.status(400).json({
          success: false,
          message: "A message cannot contain more than 4000 characters",
        });
        return;
      }

      const conversation = await getAIHealthConversationForUser(
        userId,
        conversationId,
      );

      if (!conversation) {
        res.status(404).json({
          success: false,
          message: "AI health conversation was not found",
        });
        return;
      }

      const existingMessages = getStoredAIHealthMessages(conversation.messages);
      const contextMessages: AIHealthMessage[] = [
        ...existingMessages.map(({ role, content: savedContent }) => ({
          role,
          content: savedContent,
        })),
        {
          role: "user",
          content,
        },
      ];

      const validation = normalizeAIHealthMessages(contextMessages, {
        requireLatestUser: true,
        maximumMessages: 26,
        maximumCharacters: 32000,
      });

      if (!validation.success) {
        res.status(400).json({
          success: false,
          message: validation.message,
        });
        return;
      }

      startAIHealthStream(res);
      streamStarted = true;
      writeAIHealthStatus(
        res,
        "thinking",
        "Understanding your question and previous conversation...",
      );

      const applicationContext = await buildAIHealthApplicationContext(
        currentUser,
        content,
        existingMessages,
      );

      writeAIHealthStatus(
        res,
        "tool",
        "Checking relevant SebaSathi context...",
        applicationContext.toolsUsed,
      );

      writeAIHealthStatus(
        res,
        "answering",
        "Preparing a context-aware response...",
        applicationContext.toolsUsed,
      );

      const naturalReply = await callGroqTextStream(
        [
          {
            role: "system",
            content: buildAIHealthNaturalResponsePrompt(applicationContext),
          },
          ...validation.messages,
        ],
        (delta) => {
          writeAIHealthStreamEvent(res, {
            type: "delta",
            delta,
          });
        },
        abortController.signal,
      );

      writeAIHealthStatus(
        res,
        "structuring",
        "Creating follow-up prompts, navigation actions and decision support...",
        applicationContext.toolsUsed,
      );

      let metadata: Record<string, unknown> = {};

      try {
        metadata = await callGroqAI(
          [
            {
              role: "system",
              content: buildAIHealthMetadataPrompt(
                applicationContext,
                content,
                naturalReply,
              ),
            },
            {
              role: "user",
              content: "Return the requested JSON metadata now.",
            },
          ],
          0.1,
          900,
        );
      } catch (metadataError) {
        console.error(
          "AI Health metadata generation warning:",
          metadataError instanceof Error
            ? metadataError.message
            : metadataError,
        );
      }

      const emergencyDetected = hasEmergencyWarning(validation.messages);
      const assistant = formatAIHealthAssistantResponse(
        {
          ...metadata,
          reply: naturalReply,
          toolsUsed: applicationContext.toolsUsed,
          contextMemoryUsed: applicationContext.contextMemoryUsed,
        },
        emergencyDetected,
        applicationContext,
      );

      const now = new Date();
      const userMessage: AIHealthStoredMessage = {
        id: createAIHealthMessageId(),
        role: "user",
        content,
        createdAt: now,
      };
      const assistantMessage: AIHealthStoredMessage = {
        id: createAIHealthMessageId(),
        role: "assistant",
        content: naturalReply,
        assistant,
        createdAt: new Date(),
      };

      const nextTitle =
        existingMessages.some((message) => message.role === "user") ||
        getDoctorString(conversation.title) !== "New health chat"
          ? getDoctorString(conversation.title) || "New health chat"
          : createAIHealthConversationTitle(content);

      writeAIHealthStatus(
        res,
        "saving",
        "Saving the conversation and memory...",
        applicationContext.toolsUsed,
      );

      const updatedConversation = await database
        .collection<AIHealthConversationDocument>(AI_HEALTH_CHAT_COLLECTION)
        .findOneAndUpdate(
          {
            _id: conversation._id,
          },
          {
            $set: {
              title: nextTitle,
              summaryHistoryId: null,
              summaryReport: null,
              updatedAt: assistantMessage.createdAt,
              lastMessageAt: assistantMessage.createdAt,
            },
            $push: {
              messages: {
                $each: [userMessage, assistantMessage],
              },
            },
          },
          {
            returnDocument: "after",
          },
        );

      writeAIHealthStreamEvent(res, {
        type: "result",
        data: {
          success: true,
          provider: "groq",
          model: groqModel,
          userMessage: formatAIHealthConversationMessage(userMessage),
          assistantMessage: formatAIHealthConversationMessage(assistantMessage),
          conversation: updatedConversation
            ? formatAIHealthConversation(updatedConversation)
            : null,
        },
      });

      res.end();
    } catch (error) {
      console.error("AI Health streamed chat error:", error);

      const message =
        error instanceof Error
          ? error.message
          : "Failed to receive a streamed response from the AI provider";

      if (streamStarted) {
        writeAIHealthStreamEvent(res, {
          type: "error",
          message,
        });
        res.end();
      } else {
        res.status(502).json({
          success: false,
          message,
        });
      }
    }
  },
);

/* =========================================================
   Non-streaming persistent message exchange compatibility
========================================================= */

app.post(
  "/api/v1/ai-health/conversations/:conversationId/messages",
  verifyToken,
  verifyAnyActiveUser,
  verifyAIHealthRateLimit,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const conversationId = getDoctorString(req.params.conversationId);
      const content = getDoctorString(req.body.message);

      if (!content) {
        res.status(400).json({
          success: false,
          message: "A health or application question is required",
        });
        return;
      }

      if (content.length > 4000) {
        res.status(400).json({
          success: false,
          message: "A message cannot contain more than 4000 characters",
        });
        return;
      }

      const conversation = await getAIHealthConversationForUser(
        userId,
        conversationId,
      );

      if (!conversation) {
        res.status(404).json({
          success: false,
          message: "AI health conversation was not found",
        });
        return;
      }

      const existingMessages = getStoredAIHealthMessages(conversation.messages);
      const applicationContext = await buildAIHealthApplicationContext(
        currentUser,
        content,
        existingMessages,
      );
      const contextMessages: AIHealthMessage[] = [
        ...existingMessages.map(({ role, content: savedContent }) => ({
          role,
          content: savedContent,
        })),
        {
          role: "user",
          content,
        },
      ];
      const validation = normalizeAIHealthMessages(contextMessages, {
        requireLatestUser: true,
        maximumMessages: 26,
        maximumCharacters: 32000,
      });

      if (!validation.success) {
        res.status(400).json({
          success: false,
          message: validation.message,
        });
        return;
      }

      const groqData = await callGroqAI(
        [
          {
            role: "system",
            content: `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn ONLY JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`,
          },
          ...validation.messages,
        ],
        0.2,
        1200,
      );

      const emergencyDetected = hasEmergencyWarning(validation.messages);
      const assistant = formatAIHealthAssistantResponse(
        groqData,
        emergencyDetected,
        applicationContext,
      );
      const now = new Date();
      const userMessage: AIHealthStoredMessage = {
        id: createAIHealthMessageId(),
        role: "user",
        content,
        createdAt: now,
      };
      const assistantMessage: AIHealthStoredMessage = {
        id: createAIHealthMessageId(),
        role: "assistant",
        content: assistant.reply,
        assistant,
        createdAt: new Date(),
      };
      const nextTitle =
        existingMessages.some((message) => message.role === "user") ||
        getDoctorString(conversation.title) !== "New health chat"
          ? getDoctorString(conversation.title) || "New health chat"
          : createAIHealthConversationTitle(content);

      const updatedConversation = await database
        .collection<AIHealthConversationDocument>(AI_HEALTH_CHAT_COLLECTION)
        .findOneAndUpdate(
          { _id: conversation._id },
          {
            $set: {
              title: nextTitle,
              summaryHistoryId: null,
              summaryReport: null,
              updatedAt: assistantMessage.createdAt,
              lastMessageAt: assistantMessage.createdAt,
            },
            $push: {
              messages: {
                $each: [userMessage, assistantMessage],
              },
            },
          },
          { returnDocument: "after" },
        );

      res.status(200).json({
        success: true,
        provider: "groq",
        model: groqModel,
        userMessage: formatAIHealthConversationMessage(userMessage),
        assistantMessage: formatAIHealthConversationMessage(assistantMessage),
        conversation: updatedConversation
          ? formatAIHealthConversation(updatedConversation)
          : null,
      });
    } catch (error) {
      console.error("AI Health persistent chat error:", error);

      res.status(502).json({
        success: false,
        message:
          error instanceof Error
            ? error.message
            : "Failed to receive a response from the AI provider",
      });
    }
  },
);

/* =========================================================
   Legacy AI Health chat endpoint
========================================================= */

app.post(
  "/api/v1/ai-health/chat",
  verifyToken,
  verifyAnyActiveUser,
  verifyAIHealthRateLimit,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const validation = normalizeAIHealthMessages(req.body.messages, {
        requireLatestUser: true,
        maximumMessages: 22,
        maximumCharacters: 26000,
      });

      if (!validation.success) {
        res.status(400).json({
          success: false,
          message: validation.message,
        });
        return;
      }

      const latestMessage = validation.messages.at(-1)?.content || "";
      const applicationContext = await buildAIHealthApplicationContext(
        currentUser,
        latestMessage,
        [],
      );
      const emergencyDetected = hasEmergencyWarning(validation.messages);
      const systemPrompt = `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn only JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`;

      const groqData = await callGroqAI(
        [{ role: "system", content: systemPrompt }, ...validation.messages],
        0.2,
        1200,
      );

      res.status(200).json({
        success: true,
        provider: "groq",
        model: groqModel,
        assistant: formatAIHealthAssistantResponse(
          groqData,
          emergencyDetected,
          applicationContext,
        ),
      });
    } catch (error) {
      console.error("AI Health chat error:", error);
      res.status(502).json({
        success: false,
        message:
          error instanceof Error
            ? error.message
            : "Failed to receive a response from the AI provider",
      });
    }
  },
);

/* =========================================================
   Generate and save AI Health summary
========================================================= */

app.post(
  "/api/v1/ai-health/summary",
  verifyToken,
  verifyAnyActiveUser,
  verifyAIHealthRateLimit,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const conversationId = getDoctorString(req.body.conversationId);
      let conversation: Document | null = null;
      let conversationTitle = "AI Health Conversation";
      let messagesValue: unknown = req.body.messages;

      if (conversationId) {
        conversation = await getAIHealthConversationForUser(
          userId,
          conversationId,
        );

        if (!conversation) {
          res.status(404).json({
            success: false,
            message: "AI health conversation was not found",
          });
          return;
        }

        conversationTitle =
          getDoctorString(conversation.title) || "AI Health Conversation";
        messagesValue = getStoredAIHealthMessages(conversation.messages).map(
          ({ role, content }) => ({ role, content }),
        );
      }

      const validation = normalizeAIHealthMessages(messagesValue, {
        requireLatestUser: false,
        maximumMessages: 40,
        maximumCharacters: 42000,
      });

      if (!validation.success) {
        res.status(400).json({
          success: false,
          message: validation.message,
        });
        return;
      }

      const systemPrompt = `Generate a concise structured health-conversation report for SebaSathi AI. Use only information actually present. Do not invent symptoms, duration, tests, diagnoses or medicines. Do not diagnose or prescribe. Match the user's language where practical.

Return ONLY valid JSON:
{
  "reportTitle": "short title",
  "conciseSummary": "2-3 concise sentences",
  "chiefConcerns": ["main concerns"],
  "symptoms": ["reported symptoms"],
  "durationAndPattern": "stated duration/pattern or Not clearly stated",
  "severity": "stated severity or Not clearly stated",
  "urgencyLevel": "routine | soon | urgent | emergency",
  "redFlags": ["warning signs"],
  "suggestedSpecialists": ["specialist categories"],
  "selfCareGuidance": ["low-risk general guidance"],
  "questionsForDoctor": ["useful questions"],
  "emergencyAdvice": "brief emergency advice",
  "disclaimer": "not a diagnosis or prescription"
}`;

      const groqData = await callGroqAI(
        [
          {
            role: "system",
            content: systemPrompt,
          },
          {
            role: "user",
            content: JSON.stringify(validation.messages),
          },
        ],
        0.1,
        1100,
      );

      const report = formatAIHealthSummary(groqData);
      const now = new Date();
      const userRole = getNormalizedUserRole(currentUser);
      const userName = getDoctorString(currentUser.name);
      const userEmail = normalizeDoctorEmail(currentUser.email);
      const userImage = getDoctorString(currentUser.image) || null;
      const historyCollection = database.collection(
        AI_HEALTH_HISTORY_COLLECTION,
      );

      const historyDocument = {
        conversationId: conversation ? getDoctorDocumentId(conversation) : null,
        conversationTitle,
        userId,
        userRole,
        userName,
        userEmail,
        userImage,
        patientUserId: userId,
        patientName: userName,
        patientEmail: userEmail,
        patientImage: userImage,
        provider: "groq",
        model: groqModel,
        report,
        messages: validation.messages,
        createdAt: now,
        updatedAt: now,
      };

      let history: Document | null = null;
      const existingSummaryId = getDoctorString(conversation?.summaryHistoryId);

      if (existingSummaryId) {
        const existingHistory = await historyCollection.findOne({
          $and: [
            getDoctorFilter(existingSummaryId),
            getAIHealthOwnerFilter(userId),
          ],
        });

        if (existingHistory) {
          history = await historyCollection.findOneAndUpdate(
            { _id: existingHistory._id },
            {
              $set: {
                ...historyDocument,
                createdAt: existingHistory.createdAt || now,
                updatedAt: now,
              },
            },
            { returnDocument: "after" },
          );
        }
      }

      if (!history) {
        const insertResult = await historyCollection.insertOne(historyDocument);
        history = await historyCollection.findOne({
          _id: insertResult.insertedId,
        });
      }

      if (!history) {
        throw new Error("The generated summary could not be saved");
      }

      if (conversation) {
        await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
          { _id: conversation._id },
          {
            $set: {
              summaryHistoryId: getDoctorDocumentId(history),
              summaryReport: report,
              updatedAt: now,
            },
          },
        );
      }

      res.status(201).json({
        success: true,
        message: "AI health summary generated and saved successfully",
        history: formatAIHealthHistory(history),
        conversation: conversation
          ? formatAIHealthConversation({
              ...conversation,
              summaryHistoryId: getDoctorDocumentId(history),
              summaryReport: report,
              updatedAt: now,
            })
          : null,
      });
    } catch (error) {
      console.error("AI Health summary error:", error);

      res.status(502).json({
        success: false,
        message:
          error instanceof Error
            ? error.message
            : "Failed to generate and save the AI health summary",
      });
    }
  },
);

/* =========================================================
   Patient AI Health summary history
   - Active and blocked patients can read their own history.
   - Only active patients can delete their own history.
========================================================= */

app.get(
  "/api/v1/patient/ai-health-history",
  verifyToken,
  verifyPatient,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const status = getNormalizedUserStatus(currentUser);
      const requestedPage = getPositiveInteger(req.query.page, 1, 100000);

      // Patient AI Health History always returns exactly 10 records per page.
      const limit = 10;
      const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
      const filter = getAIHealthOwnerFilter(patientUserId);
      const total = await collection.countDocuments(filter);
      const totalPages = Math.max(1, Math.ceil(total / limit));
      const page = Math.min(requestedPage, totalPages);

      const documents = await collection
        .find(filter)
        .sort({
          updatedAt: -1,
          createdAt: -1,
          _id: -1,
        })
        .skip((page - 1) * limit)
        .limit(limit)
        .toArray();

      res.status(200).json({
        success: true,
        account: {
          id: patientUserId,
          name: getDoctorString(currentUser.name),
          email: normalizeDoctorEmail(currentUser.email),
          image: getDoctorString(currentUser.image) || null,
          role: "patient",
          status,
        },
        canDelete: status === "active",
        histories: documents.map(formatAIHealthHistory),
        pagination: {
          page,
          limit,
          total,
          totalPages,
        },
      });
    } catch (error) {
      console.error("Get patient AI Health history error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve patient AI health history",
      });
    }
  },
);

app.get(
  "/api/v1/patient/ai-health-history/:historyId",
  verifyToken,
  verifyPatient,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const historyId = getDoctorString(req.params.historyId);

      if (!historyId) {
        res.status(400).json({
          success: false,
          message: "AI health history ID is required",
        });
        return;
      }

      const history = await database
        .collection(AI_HEALTH_HISTORY_COLLECTION)
        .findOne({
          $and: [
            getDoctorFilter(historyId),
            getAIHealthOwnerFilter(patientUserId),
          ],
        });

      if (!history) {
        res.status(404).json({
          success: false,
          message: "AI health history was not found",
        });
        return;
      }

      const status = getNormalizedUserStatus(currentUser);

      res.status(200).json({
        success: true,
        account: {
          id: patientUserId,
          name: getDoctorString(currentUser.name),
          email: normalizeDoctorEmail(currentUser.email),
          image: getDoctorString(currentUser.image) || null,
          role: "patient",
          status,
        },
        canDelete: status === "active",
        history: formatAIHealthHistory(history),
      });
    } catch (error) {
      console.error("Get patient AI Health history details error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to retrieve patient AI health history details",
      });
    }
  },
);

app.delete(
  "/api/v1/patient/ai-health-history/:historyId",
  verifyToken,
  verifyPatient,
  verifyActive,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const patientUserId = getDoctorDocumentId(currentUser);
      const historyId = getDoctorString(req.params.historyId);

      if (!historyId) {
        res.status(400).json({
          success: false,
          message: "AI health history ID is required",
        });
        return;
      }

      const historyCollection = database.collection(
        AI_HEALTH_HISTORY_COLLECTION,
      );

      const history = await historyCollection.findOne({
        $and: [
          getDoctorFilter(historyId),
          getAIHealthOwnerFilter(patientUserId),
        ],
      });

      if (!history) {
        res.status(404).json({
          success: false,
          message: "AI health history was not found",
        });
        return;
      }

      const deleteResult = await historyCollection.deleteOne({
        _id: history._id,
      });

      if (deleteResult.deletedCount !== 1) {
        res.status(500).json({
          success: false,
          message: "AI health history could not be deleted",
        });
        return;
      }

      const conversationId = getDoctorString(history.conversationId);

      if (conversationId) {
        await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
          {
            $and: [
              getDoctorFilter(conversationId),
              getAIHealthOwnerFilter(patientUserId),
              {
                summaryHistoryId: historyId,
              },
            ],
          },
          {
            $set: {
              summaryHistoryId: null,
              summaryReport: null,
              updatedAt: new Date(),
            },
          },
        );
      }

      res.status(200).json({
        success: true,
        message: "AI health history deleted successfully",
        deletedHistoryId: historyId,
      });
    } catch (error) {
      console.error("Delete patient AI Health history error:", error);

      res.status(500).json({
        success: false,
        message: "Failed to delete patient AI health history",
      });
    }
  },
);

/* =========================================================
   Saved AI Health summary history
========================================================= */

app.get(
  "/api/v1/ai-health/history",
  verifyToken,
  verifyAnyActiveUser,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const page = getPositiveInteger(req.query.page, 1, 100000);
      const limit = 10;
      const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
      const filter = getAIHealthOwnerFilter(userId);

      const [documents, total] = await Promise.all([
        collection
          .find(filter)
          .sort({ createdAt: -1, _id: -1 })
          .skip((page - 1) * limit)
          .limit(limit)
          .toArray(),
        collection.countDocuments(filter),
      ]);

      res.status(200).json({
        success: true,
        histories: documents.map(formatAIHealthHistory),
        pagination: {
          page,
          limit,
          total,
          totalPages: Math.max(1, Math.ceil(total / limit)),
        },
      });
    } catch (error) {
      console.error("Get AI Health history error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to retrieve AI health history",
      });
    }
  },
);

app.get(
  "/api/v1/ai-health/history/:historyId",
  verifyToken,
  verifyAnyActiveUser,
  async (req: AuthenticatedRequest, res: Response): Promise<void> => {
    try {
      if (!database) {
        res.status(503).json({
          success: false,
          message: "Database is not connected",
        });
        return;
      }

      const currentUser = await getCurrentDatabaseUser(req);

      if (!currentUser) {
        res.status(404).json({
          success: false,
          message: "User account was not found",
        });
        return;
      }

      const userId = getDoctorDocumentId(currentUser);
      const historyId = getDoctorString(req.params.historyId);
      const history = await database
        .collection(AI_HEALTH_HISTORY_COLLECTION)
        .findOne({
          $and: [getDoctorFilter(historyId), getAIHealthOwnerFilter(userId)],
        });

      if (!history) {
        res.status(404).json({
          success: false,
          message: "AI health history was not found",
        });
        return;
      }

      res.status(200).json({
        success: true,
        history: formatAIHealthHistory(history),
      });
    } catch (error) {
      console.error("Get AI Health history details error:", error);
      res.status(500).json({
        success: false,
        message: "Failed to retrieve AI health history details",
      });
    }
  },
);

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
   Start server (local development only)

   IMPORTANT (Vercel fix): On Vercel this file runs as a serverless
   function — Vercel imports the exported `app` and calls it per
   request; it never runs this block. We only call app.listen()
   locally, and we never call process.exit() on a failed DB
   connection — a failed connection just logs and lets the next
   request retry via ensureDatabaseConnection().
========================================================= */

if (!process.env.VERCEL) {
  ensureDatabaseConnection()
    .then(() => {
      app.listen(port, () => {
        console.log(
          `SebaSathi AI server is running on http://localhost:${port}`,
        );

        console.log(`JWKS URL: ${jwksUrl.toString()}`);
      });
    })
    .catch((error) => {
      console.error(
        "Unable to connect to MongoDB on startup (will keep retrying per request):",
        error instanceof Error ? error.message : error,
      );

      // Still start the HTTP server locally so it doesn't crash;
      // requests will get a 503 until the DB becomes reachable.
      app.listen(port, () => {
        console.log(
          `SebaSathi AI server is running on http://localhost:${port} (DB not yet connected)`,
        );
      });
    });
}

/* =========================================================
   Graceful shutdown (local development only)
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

if (!process.env.VERCEL) {
  process.on("SIGINT", () => {
    void shutdownServer("SIGINT");
  });

  process.on("SIGTERM", () => {
    void shutdownServer("SIGTERM");
  });
}

export default app;





// import cors from "cors";
// import dotenv from "dotenv";
// import express, {
//   type NextFunction,
//   type Request,
//   type Response,
// } from "express";
// import { createRemoteJWKSet, jwtVerify } from "jose-cjs";
// import {
//   MongoClient,
//   ObjectId,
//   ServerApiVersion,
//   type Db,
//   type Document,
//   type Filter,
// } from "mongodb";

// dotenv.config({ quiet: true });

// /* =========================================================
//    Environment variables
// ========================================================= */

// const port = Number(process.env.PORT) || 5000;

// const clientUrl = process.env.CLIENT_URL || "http://localhost:3000";

// const betterAuthUrl = (
//   process.env.BETTER_AUTH_URL || "http://localhost:3000"
// ).replace(/\/+$/, "");

// const mongoDbUri = process.env.MONGODB_URI;
// const mongoDbName = process.env.MONGODB_DB_NAME;

// const groqApiKey = process.env.GROQ_API_KEY;

// const groqApiBaseUrl = (
//   process.env.GROQ_API_BASE_URL || "https://api.groq.com/openai/v1"
// ).replace(/\/+$/, "");

// const groqModel = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

// const AI_HEALTH_HISTORY_COLLECTION = "AI-health-History";

// const AI_HEALTH_CHAT_COLLECTION = "all-history";

// if (!mongoDbUri) {
//   throw new Error("MONGODB_URI is missing from the .env file");
// }

// if (!mongoDbName) {
//   throw new Error("MONGODB_DB_NAME is missing from the .env file");
// }

// /* =========================================================
//    Express application
// ========================================================= */

// const app = express();

// /* =========================================================
//    MongoDB configuration
// ========================================================= */

// const mongoClient = new MongoClient(mongoDbUri, {
//   serverApi: {
//     version: ServerApiVersion.v1,
//     strict: false,
//     deprecationErrors: true,
//   },
// });

// let database: Db | null = null;

// /* =========================================================
//    Better Auth JWKS configuration
// ========================================================= */

// const jwksUrl = new URL(`${betterAuthUrl}/api/auth/jwks`);

// const jwks = createRemoteJWKSet(jwksUrl);

// /* =========================================================
//    Authentication types
// ========================================================= */

// type UserRole = "admin" | "doctor" | "patient";
// type UserStatus = "active" | "blocked";

// interface AuthenticatedRequest extends Request {
//   userId?: string;
//   userName?: string;
//   userEmail?: string;
//   userRole?: UserRole;
//   userStatus?: UserStatus;
// }

// /* =========================================================
//    Global middlewares
// ========================================================= */

// app.use(
//   cors({
//     origin: clientUrl,
//     credentials: true,
//   }),
// );

// app.use(
//   express.json({
//     limit: "1mb",
//   }),
// );

// app.use(express.urlencoded({ extended: true }));

// /* =========================================================
//    JWT verification middleware
// ========================================================= */

// const verifyToken = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): Promise<void> => {
//   const authorizationHeader = req.headers.authorization;

//   if (!authorizationHeader) {
//     res.status(401).json({
//       success: false,
//       message: "Authorization token is required",
//     });

//     return;
//   }

//   const [authorizationType, token] = authorizationHeader.split(" ");

//   if (authorizationType !== "Bearer" || !token) {
//     res.status(401).json({
//       success: false,
//       message: "A valid Bearer token is required",
//     });

//     return;
//   }

//   try {
//     const { payload } = await jwtVerify(token, jwks);

//     const authenticatedUserId =
//       typeof payload.sub === "string"
//         ? payload.sub
//         : typeof payload.id === "string"
//           ? payload.id
//           : undefined;

//     if (!authenticatedUserId) {
//       res.status(403).json({
//         success: false,
//         message: "Token does not contain a valid user ID",
//       });

//       return;
//     }

//     req.userId = authenticatedUserId;

//     req.userName = typeof payload.name === "string" ? payload.name : undefined;

//     req.userEmail =
//       typeof payload.email === "string" ? payload.email : undefined;

//     next();
//   } catch (error) {
//     console.error(
//       "JWT verification error:",
//       error instanceof Error ? error.message : error,
//     );

//     res.status(403).json({
//       success: false,
//       message: "Invalid or expired access token",
//     });
//   }
// };

// /* =========================================================
//    Role verification middleware
// ========================================================= */

// const verifyRole = (requiredRole: UserRole) => {
//   return async (
//     req: AuthenticatedRequest,
//     res: Response,
//     next: NextFunction,
//   ): Promise<void> => {
//     try {
//       if (!req.userId) {
//         res.status(401).json({
//           success: false,
//           message: "Authentication is required before role verification",
//         });

//         return;
//       }

//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const userQueryConditions: Record<string, unknown>[] = [
//         {
//           id: req.userId,
//         },
//       ];

//       if (req.userEmail) {
//         userQueryConditions.push({
//           email: req.userEmail,
//         });
//       }

//       const currentUser = await usersCollection.findOne({
//         $or: userQueryConditions,
//       });

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       req.userStatus = currentUser.status === "blocked" ? "blocked" : "active";

//       const currentRole = currentUser.role;

//       const validRoles: UserRole[] = ["admin", "doctor", "patient"];

//       if (
//         typeof currentRole !== "string" ||
//         !validRoles.includes(currentRole as UserRole)
//       ) {
//         res.status(403).json({
//           success: false,
//           message: "User role is missing or invalid",
//         });

//         return;
//       }

//       if (currentRole !== requiredRole) {
//         res.status(403).json({
//           success: false,
//           message: `${requiredRole} access is required`,
//         });

//         return;
//       }

//       req.userRole = currentRole as UserRole;

//       next();
//     } catch (error) {
//       console.error(
//         "Role verification error:",
//         error instanceof Error ? error.message : error,
//       );

//       res.status(500).json({
//         success: false,
//         message: "Failed to verify current user role",
//       });
//     }
//   };
// };

// /* =========================================================
//    Admin, doctor and patient middlewares
// ========================================================= */

// const verifyAdmin = verifyRole("admin");

// const verifyDoctor = verifyRole("doctor");

// const verifyPatient = verifyRole("patient");

// /**
//  * Allows blocked users to read data, but prevents them
//  * from creating, editing, deleting, or changing status.
//  *
//  * Always use this after verifyRole().
//  */
// const verifyActive = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): void => {
//   if (req.userStatus !== "active") {
//     res.status(403).json({
//       success: false,
//       message:
//         "Your account is blocked. You can view data, but you cannot perform this action.",
//       code: "READ_ONLY_ACCOUNT",
//     });

//     return;
//   }

//   next();
// };

// /**
//  * Allows any authenticated role (admin, doctor or patient)
//  * to use protected features when the account status is active.
//  */
// const verifyAnyActiveUser = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): Promise<void> => {
//   try {
//     if (!req.userId) {
//       res.status(401).json({
//         success: false,
//         message: "Authentication is required",
//       });

//       return;
//     }

//     if (!database) {
//       res.status(503).json({
//         success: false,
//         message: "Database is not connected",
//       });

//       return;
//     }

//     const userQueryConditions: Record<string, unknown>[] = [
//       {
//         id: req.userId,
//       },
//     ];

//     if (req.userEmail) {
//       userQueryConditions.push({
//         email: req.userEmail.toLowerCase(),
//       });
//     }

//     const currentUser = await database.collection("user").findOne({
//       $or: userQueryConditions,
//     });

//     if (!currentUser) {
//       res.status(404).json({
//         success: false,
//         message: "User account was not found",
//       });

//       return;
//     }

//     const currentRole = currentUser.role;
//     const validRoles: UserRole[] = ["admin", "doctor", "patient"];

//     if (
//       typeof currentRole !== "string" ||
//       !validRoles.includes(currentRole as UserRole)
//     ) {
//       res.status(403).json({
//         success: false,
//         message: "User role is missing or invalid",
//       });

//       return;
//     }

//     const currentStatus: UserStatus =
//       currentUser.status === "blocked" ? "blocked" : "active";

//     req.userRole = currentRole as UserRole;
//     req.userStatus = currentStatus;

//     if (currentStatus !== "active") {
//       res.status(403).json({
//         success: false,
//         message:
//           "Your account is blocked. Only active accounts can use the AI Health Assistant.",
//         code: "READ_ONLY_ACCOUNT",
//       });

//       return;
//     }

//     next();
//   } catch (error) {
//     console.error(
//       "Active user verification error:",
//       error instanceof Error ? error.message : error,
//     );

//     res.status(500).json({
//       success: false,
//       message: "Failed to verify active user account",
//     });
//   }
// };

// /* =========================================================
//    Public root route
// ========================================================= */

// app.get("/", (_req: Request, res: Response) => {
//   res.status(200).json({
//     success: true,
//     message: "SebaSathi AI server is running",
//   });
// });

// /* =========================================================
//    Public health route
// ========================================================= */

// app.get("/api/v1/health", async (_req: Request, res: Response) => {
//   try {
//     if (!database) {
//       res.status(503).json({
//         success: false,
//         message: "Database is not connected",
//       });

//       return;
//     }

//     await database.command({ ping: 1 });

//     res.status(200).json({
//       success: true,
//       message: "SebaSathi AI API is healthy",
//       database: "connected",
//       databaseName: database.databaseName,
//       timestamp: new Date().toISOString(),
//     });
//   } catch {
//     res.status(503).json({
//       success: false,
//       message: "MongoDB connection is unavailable",
//       database: "disconnected",
//       timestamp: new Date().toISOString(),
//     });
//   }
// });

// /* =========================================================
//    Protected authentication test route
// ========================================================= */

// app.get(
//   "/api/v1/auth/me",
//   verifyToken,
//   (req: AuthenticatedRequest, res: Response) => {
//     res.status(200).json({
//       success: true,
//       message: "Authenticated user retrieved successfully",
//       user: {
//         id: req.userId,
//         name: req.userName || null,
//         email: req.userEmail || null,
//       },
//     });
//   },
// );

// /*
//   Admin API middleware:

//   app.get(
//     "/api/v1/admin/your-api",
//     verifyToken,
//     verifyAdmin,
//     yourController
//   );
// */

// /*
//   Doctor API middleware:

//   app.get(
//     "/api/v1/doctor/your-api",
//     verifyToken,
//     verifyDoctor,
//     yourController
//   );
// */

// /*
//   Patient API middleware:

//   app.get(
//     "/api/v1/patient/your-api",
//     verifyToken,
//     verifyPatient,
//     yourController
//   );
// */

// /* =========================================================
//    Current authenticated user
// ========================================================= */

// app.get(
//   "/api/users/current",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       if (!req.userId) {
//         res.status(401).json({
//           success: false,
//           message: "Authenticated user ID was not found",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const userQueryConditions: Record<string, unknown>[] = [
//         {
//           id: req.userId,
//         },
//       ];

//       if (req.userEmail) {
//         userQueryConditions.push({
//           email: req.userEmail.toLowerCase(),
//         });
//       }

//       const currentUser = await usersCollection.findOne({
//         $or: userQueryConditions,
//       });

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const currentRole: UserRole =
//         currentUser.role === "admin" ||
//         currentUser.role === "doctor" ||
//         currentUser.role === "patient"
//           ? currentUser.role
//           : "patient";

//       const currentStatus: "active" | "blocked" =
//         currentUser.status === "blocked" ? "blocked" : "active";

//       const currentUserId =
//         typeof currentUser.id === "string" && currentUser.id.trim()
//           ? currentUser.id
//           : currentUser._id instanceof ObjectId
//             ? currentUser._id.toHexString()
//             : req.userId;

//       res.status(200).json({
//         id: currentUserId,
//         _id: currentUserId,
//         name: typeof currentUser.name === "string" ? currentUser.name : null,
//         email: typeof currentUser.email === "string" ? currentUser.email : null,
//         image: typeof currentUser.image === "string" ? currentUser.image : null,
//         role: currentRole,
//         status: currentStatus,
//       });
//     } catch (error) {
//       console.error(
//         "Get current user error:",
//         error instanceof Error ? error.message : error,
//       );

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve current user",
//       });
//     }
//   },
// );

// /* =========================================================
//    Manage Doctors helpers
// ========================================================= */

// type DoctorStatus = "active" | "blocked";

// const getDoctorString = (value: unknown): string => {
//   return typeof value === "string" ? value.trim() : "";
// };

// const getDoctorNumber = (value: unknown): number => {
//   const numberValue =
//     typeof value === "number"
//       ? value
//       : typeof value === "string" && value.trim()
//         ? Number(value)
//         : 0;

//   return Number.isFinite(numberValue) ? Math.max(0, numberValue) : 0;
// };

// const normalizeDoctorEmail = (value: unknown): string => {
//   return getDoctorString(value).toLowerCase();
// };

// const isValidDoctorEmail = (email: string): boolean => {
//   return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
// };

// const escapeDoctorSearch = (value: string): string => {
//   return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// };

// const getDoctorDocumentId = (document: Document): string => {
//   if (typeof document.id === "string" && document.id.trim()) {
//     return document.id;
//   }

//   if (document._id instanceof ObjectId) {
//     return document._id.toHexString();
//   }

//   return String(document._id || "");
// };

// const getDoctorFilter = (doctorId: string): Filter<Document> => {
//   const conditions: Filter<Document>[] = [
//     {
//       id: doctorId,
//     },
//   ];

//   if (ObjectId.isValid(doctorId)) {
//     conditions.push({
//       _id: new ObjectId(doctorId),
//     });
//   }

//   return {
//     $or: conditions,
//   };
// };

// const getUserFilter = (userId: string): Filter<Document> => {
//   const conditions: Filter<Document>[] = [
//     {
//       id: userId,
//     },
//   ];

//   if (ObjectId.isValid(userId)) {
//     conditions.push({
//       _id: new ObjectId(userId),
//     });
//   }

//   return {
//     $or: conditions,
//   };
// };

// const formatDoctorDate = (value: unknown): string | null => {
//   if (value instanceof Date) {
//     return value.toISOString();
//   }

//   if (typeof value === "string" || typeof value === "number") {
//     const date = new Date(value);

//     return Number.isNaN(date.getTime()) ? null : date.toISOString();
//   }

//   return null;
// };

// const formatDoctor = (doctor: Document) => {
//   return {
//     id: getDoctorDocumentId(doctor),

//     userId: typeof doctor.userId === "string" ? doctor.userId : "",

//     name: getDoctorString(doctor.name),

//     email: normalizeDoctorEmail(doctor.email),

//     image: getDoctorString(doctor.image) || null,

//     phone: getDoctorString(doctor.phone),

//     specialization: getDoctorString(doctor.specialization),

//     qualification: getDoctorString(doctor.qualification),

//     experienceYears: getDoctorNumber(doctor.experienceYears),

//     hospital:
//       getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),

//     address: getDoctorString(doctor.address),

//     bio: getDoctorString(doctor.bio),

//     role: "doctor" as const,

//     status:
//       doctor.status === "blocked" ? ("blocked" as const) : ("active" as const),

//     createdAt: formatDoctorDate(doctor.createdAt),

//     updatedAt: formatDoctorDate(doctor.updatedAt),
//   };
// };

// const readBetterAuthResponse = async (
//   response: globalThis.Response,
// ): Promise<unknown> => {
//   try {
//     return await response.json();
//   } catch {
//     return null;
//   }
// };

// const getBetterAuthError = (value: unknown): string => {
//   if (typeof value !== "object" || value === null) {
//     return "Doctor authentication account could not be created";
//   }

//   const data = value as Record<string, unknown>;

//   if (typeof data.message === "string" && data.message.trim()) {
//     return data.message;
//   }

//   if (typeof data.error === "object" && data.error !== null) {
//     const error = data.error as Record<string, unknown>;

//     if (typeof error.message === "string" && error.message.trim()) {
//       return error.message;
//     }
//   }

//   return "Doctor authentication account could not be created";
// };

// /* =========================================================
//    Admin patient management helpers
// ========================================================= */

// type ManagedPatientStatus = "active" | "blocked";

// const getAdminPatientFilter = (patientId: string): Filter<Document> => {
//   return {
//     $and: [getUserFilter(patientId), { role: "patient" }],
//   };
// };

// const formatManagedPatient = (patient: Document) => {
//   const status: ManagedPatientStatus =
//     patient.status === "blocked" ? "blocked" : "active";

//   return {
//     id: getDoctorDocumentId(patient),
//     name: getDoctorString(patient.name),
//     email: normalizeDoctorEmail(patient.email),
//     image: getDoctorString(patient.image) || null,
//     role: "patient" as const,
//     status,
//     emailVerified: patient.emailVerified === true,
//     phone: getDoctorString(patient.phone) || null,
//     address: getDoctorString(patient.address) || null,
//     dateOfBirth: getDoctorString(patient.dateOfBirth) || null,
//     gender: getDoctorString(patient.gender) || null,
//     bloodGroup: getDoctorString(patient.bloodGroup) || null,
//     occupation: getDoctorString(patient.occupation) || null,
//     city: getDoctorString(patient.city) || null,
//     country: getDoctorString(patient.country) || null,
//     bio: getDoctorString(patient.bio) || null,
//     emergencyContactName: getDoctorString(patient.emergencyContactName) || null,
//     emergencyContactPhone:
//       getDoctorString(patient.emergencyContactPhone) ||
//       getDoctorString(patient.emergencyContact) ||
//       null,
//     createdAt: formatDoctorDate(patient.createdAt),
//     updatedAt: formatDoctorDate(patient.updatedAt),
//   };
// };

// /* =========================================================
//    GET managed patients (10 per page)
// ========================================================= */

// app.get(
//   "/api/v1/admin/patients",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const search = getDoctorString(req.query.search);
//       const requestedStatus = getDoctorString(req.query.status);
//       const requestedPage = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;

//       const conditions: Filter<Document>[] = [{ role: "patient" }];

//       if (requestedStatus === "active" || requestedStatus === "blocked") {
//         conditions.push({ status: requestedStatus });
//       }

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         conditions.push({
//           $or: [
//             {
//               name: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               email: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       const filter: Filter<Document> = { $and: conditions };
//       const usersCollection = database.collection("user");
//       const total = await usersCollection.countDocuments(filter);
//       const totalPages = Math.max(1, Math.ceil(total / limit));
//       const page = Math.min(requestedPage, totalPages);

//       const patientDocuments = await usersCollection
//         .find(filter)
//         .sort({
//           updatedAt: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .skip((page - 1) * limit)
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         patients: patientDocuments.map(formatManagedPatient),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages,
//         },
//       });
//     } catch (error) {
//       console.error("Get managed patients error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patients",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET managed patient details
// ========================================================= */

// app.get(
//   "/api/v1/admin/patients/:patientId",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);

//       if (!patientId) {
//         res.status(400).json({
//           success: false,
//           message: "Patient ID is required",
//         });
//         return;
//       }

//       const patient = await database
//         .collection("user")
//         .findOne(getAdminPatientFilter(patientId));

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         patient: formatManagedPatient(patient),
//       });
//     } catch (error) {
//       console.error("Get managed patient details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient details",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH block or activate patient
// ========================================================= */

// app.patch(
//   "/api/v1/admin/patients/:patientId/status",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);
//       const requestedStatus = getDoctorString(req.body.status);

//       if (requestedStatus !== "active" && requestedStatus !== "blocked") {
//         res.status(400).json({
//           success: false,
//           message: "Status must be active or blocked",
//         });
//         return;
//       }

//       const usersCollection = database.collection("user");
//       const patient = await usersCollection.findOne(
//         getAdminPatientFilter(patientId),
//       );

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       const status = requestedStatus as ManagedPatientStatus;
//       const updatedPatient = await usersCollection.findOneAndUpdate(
//         { _id: patient._id },
//         {
//           $set: {
//             status,
//             updatedAt: new Date(),
//           },
//         },
//         { returnDocument: "after" },
//       );

//       if (!updatedPatient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       if (status === "blocked") {
//         await database.collection("session").deleteMany({
//           userId: getDoctorDocumentId(patient),
//         });
//       }

//       res.status(200).json({
//         success: true,
//         message:
//           status === "blocked"
//             ? "Patient blocked successfully"
//             : "Patient activated successfully",
//         patient: formatManagedPatient(updatedPatient),
//       });
//     } catch (error) {
//       console.error("Change patient status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to change patient status",
//       });
//     }
//   },
// );

// /* =========================================================
//    DELETE patient account
// ========================================================= */

// app.delete(
//   "/api/v1/admin/patients/:patientId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);

//       if (!patientId) {
//         res.status(400).json({
//           success: false,
//           message: "Patient ID is required",
//         });
//         return;
//       }

//       const usersCollection = database.collection("user");
//       const patient = await usersCollection.findOne(
//         getAdminPatientFilter(patientId),
//       );

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(patient);
//       const email = normalizeDoctorEmail(patient.email);

//       await Promise.all([
//         database.collection("session").deleteMany({ userId }),
//         database.collection("account").deleteMany({ userId }),
//         database.collection("verification").deleteMany({
//           $or: [{ identifier: email }, { value: email }],
//         }),
//       ]);

//       const deleteResult = await usersCollection.deleteOne({
//         _id: patient._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Patient account could not be deleted",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Patient account deleted successfully",
//         deletedPatientId: userId,
//       });
//     } catch (error) {
//       console.error("Delete patient account error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete patient account",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET all doctors
// ========================================================= */

// app.get(
//   "/api/v1/admin/doctors",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const search = getDoctorString(req.query.search);

//       const status = getDoctorString(req.query.status);

//       const page = Math.max(
//         1,
//         Math.floor(getDoctorNumber(req.query.page) || 1),
//       );

//       const limit = Math.min(
//         100,
//         Math.max(1, Math.floor(getDoctorNumber(req.query.limit) || 50)),
//       );

//       const filter: Filter<Document> = {};

//       if (status === "active" || status === "blocked") {
//         filter.status = status;
//       }

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         filter.$or = [
//           {
//             name: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             email: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             phone: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             specialization: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//         ];
//       }

//       const [doctorDocuments, total] = await Promise.all([
//         doctorsCollection
//           .find(filter)
//           .sort({
//             createdAt: -1,
//             _id: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         doctorsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,

//         doctors: doctorDocuments.map(formatDoctor),

//         pagination: {
//           page,
//           limit,
//           total,

//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get doctors error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET single doctor details
// ========================================================= */

// app.get(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       if (!doctorId) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         doctor: formatDoctor(doctor),
//       });
//     } catch (error) {
//       console.error("Get doctor details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor details",
//       });
//     }
//   },
// );

// /* =========================================================
//    POST create doctor
// ========================================================= */

// app.post(
//   "/api/v1/admin/doctors",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const name = getDoctorString(req.body.name);

//       const email = normalizeDoctorEmail(req.body.email);

//       const password = getDoctorString(req.body.password);

//       const specialization = getDoctorString(req.body.specialization);

//       if (!name || !email || !password || !specialization) {
//         res.status(400).json({
//           success: false,
//           message: "Name, email, password and specialization are required",
//         });

//         return;
//       }

//       if (!isValidDoctorEmail(email)) {
//         res.status(400).json({
//           success: false,
//           message: "A valid email address is required",
//         });

//         return;
//       }

//       if (password.length < 8) {
//         res.status(400).json({
//           success: false,
//           message: "Password must contain at least 8 characters",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const doctorsCollection = database.collection("doctors");

//       const accountsCollection = database.collection("account");

//       const sessionsCollection = database.collection("session");

//       const existingUser = await usersCollection.findOne({
//         email,
//       });

//       const existingDoctor = await doctorsCollection.findOne({
//         email,
//       });

//       if (existingUser || existingDoctor) {
//         res.status(409).json({
//           success: false,
//           message: "An account with this email already exists",
//         });

//         return;
//       }

//       /*
//        * Better Auth securely creates the email/password account.
//        * Raw password MongoDB-তে save হবে না।
//        */
//       const signupResponse = await fetch(
//         `${betterAuthUrl}/api/auth/sign-up/email`,
//         {
//           method: "POST",

//           headers: {
//             "content-type": "application/json",
//             accept: "application/json",
//             origin: betterAuthUrl,
//           },

//           body: JSON.stringify({
//             name,
//             email,
//             password,
//           }),
//         },
//       );

//       const signupData = await readBetterAuthResponse(signupResponse);

//       if (!signupResponse.ok) {
//         res
//           .status(signupResponse.status >= 500 ? 502 : signupResponse.status)
//           .json({
//             success: false,
//             message: getBetterAuthError(signupData),
//           });

//         return;
//       }

//       const createdUser = await usersCollection.findOne({
//         email,
//       });

//       if (!createdUser) {
//         res.status(500).json({
//           success: false,
//           message: "Authentication account was created but user was not found",
//         });

//         return;
//       }

//       const userId = getDoctorDocumentId(createdUser);

//       const now = new Date();

//       await usersCollection.updateOne(
//         {
//           _id: createdUser._id,
//         },
//         {
//           $set: {
//             name,
//             email,
//             role: "doctor",
//             status: "active",
//             updatedAt: now,
//           },
//         },
//       );

//       /*
//        * Admin-created doctor will sign in manually.
//        * Remove any session created by signup.
//        */
//       await sessionsCollection.deleteMany({
//         userId,
//       });

//       const doctorDocument = {
//         userId,

//         name,
//         email,

//         image: getDoctorString(req.body.image) || null,

//         phone: getDoctorString(req.body.phone),

//         specialization,

//         qualification: getDoctorString(req.body.qualification),

//         experienceYears: getDoctorNumber(req.body.experienceYears),

//         hospital: getDoctorString(req.body.hospital),

//         address: getDoctorString(req.body.address),

//         bio: getDoctorString(req.body.bio),

//         role: "doctor" as const,

//         status: "active" as const,

//         createdAt: now,
//         updatedAt: now,
//       };

//       try {
//         const insertResult = await doctorsCollection.insertOne(doctorDocument);

//         const createdDoctor = await doctorsCollection.findOne({
//           _id: insertResult.insertedId,
//         });

//         if (!createdDoctor) {
//           throw new Error("Created doctor profile was not found");
//         }

//         res.status(201).json({
//           success: true,
//           message: "Doctor created successfully",
//           doctor: formatDoctor(createdDoctor),
//         });
//       } catch (profileError) {
//         /*
//          * Roll back authentication data if
//          * doctor profile creation fails.
//          */
//         await Promise.all([
//           sessionsCollection.deleteMany({
//             userId,
//           }),

//           accountsCollection.deleteMany({
//             userId,
//           }),

//           usersCollection.deleteOne({
//             _id: createdUser._id,
//           }),
//         ]);

//         throw profileError;
//       }
//     } catch (error) {
//       console.error("Create doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to create doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH edit doctor
// ========================================================= */

// app.patch(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const name = getDoctorString(req.body.name);

//       const email = normalizeDoctorEmail(req.body.email);

//       const specialization = getDoctorString(req.body.specialization);

//       if (!doctorId || !name || !email || !specialization) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID, name, email and specialization are required",
//         });

//         return;
//       }

//       if (!isValidDoctorEmail(email)) {
//         res.status(400).json({
//           success: false,
//           message: "A valid email address is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       const linkedUser = userId
//         ? await usersCollection.findOne(getUserFilter(userId))
//         : null;

//       const duplicateDoctor = await doctorsCollection.findOne({
//         email,

//         _id: {
//           $ne: doctor._id,
//         },
//       });

//       const duplicateUser = await usersCollection.findOne({
//         email,

//         ...(linkedUser
//           ? {
//               _id: {
//                 $ne: linkedUser._id,
//               },
//             }
//           : {}),
//       });

//       if (duplicateDoctor || duplicateUser) {
//         res.status(409).json({
//           success: false,
//           message: "Another account already uses this email",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedDoctor = await doctorsCollection.findOneAndUpdate(
//         {
//           _id: doctor._id,
//         },
//         {
//           $set: {
//             name,
//             email,

//             image: getDoctorString(req.body.image) || null,

//             phone: getDoctorString(req.body.phone),

//             specialization,

//             qualification: getDoctorString(req.body.qualification),

//             experienceYears: getDoctorNumber(req.body.experienceYears),

//             hospital: getDoctorString(req.body.hospital),

//             address: getDoctorString(req.body.address),

//             bio: getDoctorString(req.body.bio),

//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       if (!updatedDoctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       if (linkedUser) {
//         await usersCollection.updateOne(
//           {
//             _id: linkedUser._id,
//           },
//           {
//             $set: {
//               name,
//               email,

//               image: getDoctorString(req.body.image) || null,

//               updatedAt: now,
//             },
//           },
//         );
//       }

//       res.status(200).json({
//         success: true,
//         message: "Doctor updated successfully",
//         doctor: formatDoctor(updatedDoctor),
//       });
//     } catch (error) {
//       console.error("Update doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH block or activate doctor
// ========================================================= */

// app.patch(
//   "/api/v1/admin/doctors/:doctorId/status",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const requestedStatus = getDoctorString(req.body.status);

//       if (requestedStatus !== "active" && requestedStatus !== "blocked") {
//         res.status(400).json({
//           success: false,
//           message: "Status must be active or blocked",
//         });

//         return;
//       }

//       const status = requestedStatus as DoctorStatus;

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedDoctor = await doctorsCollection.findOneAndUpdate(
//         {
//           _id: doctor._id,
//         },
//         {
//           $set: {
//             status,
//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       if (!updatedDoctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       if (userId) {
//         await usersCollection.updateOne(getUserFilter(userId), {
//           $set: {
//             status,
//             updatedAt: now,
//           },
//         });
//       }

//       res.status(200).json({
//         success: true,

//         message:
//           status === "blocked"
//             ? "Doctor blocked successfully"
//             : "Doctor activated successfully",

//         doctor: formatDoctor(updatedDoctor),
//       });
//     } catch (error) {
//       console.error("Change doctor status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to change doctor status",
//       });
//     }
//   },
// );

// /* =========================================================
//    DELETE doctor
// ========================================================= */

// app.delete(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       if (!doctorId) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const accountsCollection = database.collection("account");

//       const sessionsCollection = database.collection("session");

//       const verificationCollection = database.collection("verification");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       if (userId) {
//         await Promise.all([
//           sessionsCollection.deleteMany({
//             userId,
//           }),

//           accountsCollection.deleteMany({
//             userId,
//           }),

//           verificationCollection.deleteMany({
//             $or: [
//               {
//                 identifier: doctor.email,
//               },
//               {
//                 value: doctor.email,
//               },
//             ],
//           }),
//         ]);

//         await usersCollection.deleteOne(getUserFilter(userId));
//       }

//       const deleteResult = await doctorsCollection.deleteOne({
//         _id: doctor._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Doctor could not be deleted",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Doctor deleted successfully",
//         deletedDoctorId: getDoctorDocumentId(doctor),
//       });
//     } catch (error) {
//       console.error("Delete doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctors, appointments and reviews
// ========================================================= */

// type AppointmentStatus = "pending" | "approved" | "completed" | "rejected";

// const ACTIVE_APPOINTMENT_STATUSES: AppointmentStatus[] = [
//   "pending",
//   "approved",
// ];

// const getPositiveInteger = (
//   value: unknown,
//   fallback: number,
//   maximum: number,
// ): number => {
//   const parsed = Number(value);

//   if (!Number.isFinite(parsed)) {
//     return fallback;
//   }

//   return Math.min(maximum, Math.max(1, Math.floor(parsed)));
// };

// const getCurrentDatabaseUser = async (
//   req: AuthenticatedRequest,
// ): Promise<Document | null> => {
//   if (!database || !req.userId) {
//     return null;
//   }

//   const conditions: Filter<Document>[] = [getUserFilter(req.userId)];

//   if (req.userEmail) {
//     conditions.push({
//       email: req.userEmail.toLowerCase(),
//     });
//   }

//   return database.collection("user").findOne({
//     $or: conditions,
//   });
// };

// const getNormalizedUserRole = (user: Document): UserRole => {
//   return user.role === "admin" ||
//     user.role === "doctor" ||
//     user.role === "patient"
//     ? user.role
//     : "patient";
// };

// const getNormalizedUserStatus = (user: Document): UserStatus => {
//   return user.status === "blocked" ? "blocked" : "active";
// };

// const getPublicDoctor = (doctor: Document) => {
//   const ratingAverage = Number(doctor.ratingAverage);

//   const ratingCount = Number(doctor.ratingCount);

//   return {
//     ...formatDoctor(doctor),

//     ratingAverage: Number.isFinite(ratingAverage)
//       ? Number(ratingAverage.toFixed(1))
//       : 0,

//     ratingCount: Number.isFinite(ratingCount)
//       ? Math.max(0, Math.floor(ratingCount))
//       : 0,
//   };
// };

// const getReviewDocumentId = (document: Document): string => {
//   return getDoctorDocumentId(document);
// };

// const formatReview = (review: Document) => {
//   return {
//     id: getReviewDocumentId(review),
//     doctorId: getDoctorString(review.doctorId),
//     userId: getDoctorString(review.userId),
//     userName: getDoctorString(review.userName),
//     userEmail: normalizeDoctorEmail(review.userEmail),
//     userImage: getDoctorString(review.userImage) || null,
//     rating: Math.min(
//       5,
//       Math.max(1, Math.floor(getDoctorNumber(review.rating))),
//     ),
//     review: getDoctorString(review.review),
//     createdAt: formatDoctorDate(review.createdAt),
//     updatedAt: formatDoctorDate(review.updatedAt),
//   };
// };

// const refreshDoctorRatingStats = async (doctorId: string): Promise<void> => {
//   if (!database) {
//     return;
//   }

//   const reviewsCollection = database.collection("reviews");

//   const doctorsCollection = database.collection("doctors");

//   const [stats] = await reviewsCollection
//     .aggregate([
//       {
//         $match: {
//           doctorId,
//         },
//       },
//       {
//         $group: {
//           _id: "$doctorId",
//           ratingAverage: {
//             $avg: "$rating",
//           },
//           ratingCount: {
//             $sum: 1,
//           },
//         },
//       },
//     ])
//     .toArray();

//   await doctorsCollection.updateOne(getDoctorFilter(doctorId), {
//     $set: {
//       ratingAverage:
//         typeof stats?.ratingAverage === "number"
//           ? Number(stats.ratingAverage.toFixed(2))
//           : 0,
//       ratingCount:
//         typeof stats?.ratingCount === "number" ? stats.ratingCount : 0,
//       updatedAt: new Date(),
//     },
//   });
// };

// const formatAppointment = (appointment: Document) => {
//   return {
//     id: getDoctorDocumentId(appointment),
//     doctorId: getDoctorString(appointment.doctorId),
//     doctorUserId: getDoctorString(appointment.doctorUserId),
//     doctorName: getDoctorString(appointment.doctorName),
//     doctorImage: getDoctorString(appointment.doctorImage) || null,
//     specialization: getDoctorString(appointment.specialization),
//     hospital: getDoctorString(appointment.hospital),
//     patientUserId: getDoctorString(appointment.patientUserId),
//     patientName: getDoctorString(appointment.patientName),
//     patientEmail: normalizeDoctorEmail(appointment.patientEmail),
//     patientImage: getDoctorString(appointment.patientImage) || null,
//     phone: getDoctorString(appointment.phone),
//     address: getDoctorString(appointment.address),
//     problemTitle: getDoctorString(appointment.problemTitle),
//     symptomsDescription: getDoctorString(appointment.symptomsDescription),
//     appointmentDate: getDoctorString(appointment.appointmentDate),
//     appointmentTime: getDoctorString(appointment.appointmentTime),
//     status:
//       appointment.status === "approved" ||
//       appointment.status === "completed" ||
//       appointment.status === "rejected"
//         ? appointment.status
//         : "pending",
//     rejectionReason: getDoctorString(appointment.rejectionReason) || null,
//     approvedAt: formatDoctorDate(appointment.approvedAt),
//     completedAt: formatDoctorDate(appointment.completedAt),
//     rejectedAt: formatDoctorDate(appointment.rejectedAt),
//     rescheduledAt: formatDoctorDate(appointment.rescheduledAt),
//     rescheduledBy: getDoctorString(appointment.rescheduledBy) || null,
//     rescheduleReason: getDoctorString(appointment.rescheduleReason) || null,
//     createdAt: formatDoctorDate(appointment.createdAt),
//     updatedAt: formatDoctorDate(appointment.updatedAt),
//   };
// };

// const attachPatientImages = async (
//   appointments: Document[],
// ): Promise<Document[]> => {
//   if (!database || appointments.length === 0) {
//     return appointments;
//   }

//   const patientUserIds = Array.from(
//     new Set(
//       appointments
//         .map((appointment) => getDoctorString(appointment.patientUserId))
//         .filter(Boolean),
//     ),
//   );

//   if (patientUserIds.length === 0) {
//     return appointments;
//   }

//   const objectIds = patientUserIds
//     .filter((userId) => ObjectId.isValid(userId))
//     .map((userId) => new ObjectId(userId));

//   const userConditions: Filter<Document>[] = [
//     {
//       id: {
//         $in: patientUserIds,
//       },
//     },
//   ];

//   if (objectIds.length > 0) {
//     userConditions.push({
//       _id: {
//         $in: objectIds,
//       },
//     });
//   }

//   const users = await database
//     .collection("user")
//     .find(
//       {
//         $or: userConditions,
//       },
//       {
//         projection: {
//           id: 1,
//           image: 1,
//         },
//       },
//     )
//     .toArray();

//   const imageByUserId = new Map<string, string | null>();

//   users.forEach((user) => {
//     imageByUserId.set(
//       getDoctorDocumentId(user),
//       getDoctorString(user.image) || null,
//     );
//   });

//   return appointments.map((appointment) => ({
//     ...appointment,
//     patientImage:
//       getDoctorString(appointment.patientImage) ||
//       imageByUserId.get(getDoctorString(appointment.patientUserId)) ||
//       null,
//   }));
// };

// /* =========================================================
//    Public doctor filters
// ========================================================= */

// app.get(
//   "/api/v1/doctors/filters",
//   async (_req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorDocuments = await database
//         .collection("doctors")
//         .find(
//           {
//             status: "active",
//           },
//           {
//             projection: {
//               specialization: 1,
//               qualification: 1,
//               experienceYears: 1,
//               hospital: 1,
//               chamber: 1,
//             },
//           },
//         )
//         .toArray();

//       const specializations = new Set<string>();
//       const qualifications = new Set<string>();
//       const hospitals = new Set<string>();
//       const experienceYears = new Set<number>();

//       doctorDocuments.forEach((doctor) => {
//         const specialization = getDoctorString(doctor.specialization);
//         const qualification = getDoctorString(doctor.qualification);
//         const hospital =
//           getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber);
//         const experience = getDoctorNumber(doctor.experienceYears);

//         if (specialization) {
//           specializations.add(specialization);
//         }

//         if (qualification) {
//           qualifications.add(qualification);
//         }

//         if (hospital) {
//           hospitals.add(hospital);
//         }

//         experienceYears.add(experience);
//       });

//       res.status(200).json({
//         success: true,
//         filters: {
//           specializations: Array.from(specializations).sort((a, b) =>
//             a.localeCompare(b),
//           ),
//           qualifications: Array.from(qualifications).sort((a, b) =>
//             a.localeCompare(b),
//           ),
//           hospitals: Array.from(hospitals).sort((a, b) => a.localeCompare(b)),
//           experienceYears: Array.from(experienceYears).sort((a, b) => a - b),
//         },
//       });
//     } catch (error) {
//       console.error("Get public doctor filters error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor filters",
//       });
//     }
//   },
// );

// /* =========================================================
//    Top Rated Public Doctors
// ========================================================= */

// app.get(
//   "/api/v1/doctors/top-rated",
//   async (_req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const doctors = await doctorsCollection
//         .find({
//           status: "active",
//         })
//         .sort({
//           ratingAverage: -1,
//           ratingCount: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .limit(4)
//         .toArray();

//       res.status(200).json({
//         success: true,

//         doctors: doctors.map(getPublicDoctor),
//       });
//     } catch (error) {
//       console.error("Get top rated doctors error:", error);

//       res.status(500).json({
//         success: false,

//         message: "Failed to retrieve top rated doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctor list
// ========================================================= */

// app.get(
//   "/api/v1/doctors",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const search = getDoctorString(req.query.search);
//       const specialization = getDoctorString(req.query.specialization);
//       const qualification = getDoctorString(req.query.qualification);
//       const hospital = getDoctorString(req.query.hospital);
//       const experienceValue = getDoctorString(req.query.experienceYears);

//       const page = getPositiveInteger(req.query.page, 1, 100000);

//       const limit = 8;

//       const conditions: Filter<Document>[] = [
//         {
//           status: "active",
//         },
//       ];

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         conditions.push({
//           $or: [
//             {
//               name: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               specialization: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               qualification: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       if (specialization) {
//         conditions.push({
//           specialization: {
//             $regex: `^${escapeDoctorSearch(specialization)}$`,
//             $options: "i",
//           },
//         });
//       }

//       if (qualification) {
//         conditions.push({
//           qualification: {
//             $regex: `^${escapeDoctorSearch(qualification)}$`,
//             $options: "i",
//           },
//         });
//       }

//       if (hospital) {
//         const safeHospital = `^${escapeDoctorSearch(hospital)}$`;

//         conditions.push({
//           $or: [
//             {
//               hospital: {
//                 $regex: safeHospital,
//                 $options: "i",
//               },
//             },
//             {
//               chamber: {
//                 $regex: safeHospital,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       if (experienceValue) {
//         const experienceYears = Number(experienceValue);

//         if (Number.isFinite(experienceYears)) {
//           conditions.push({
//             experienceYears: Math.max(0, Math.floor(experienceYears)),
//           });
//         }
//       }

//       const filter: Filter<Document> = {
//         $and: conditions,
//       };

//       const doctorsCollection = database.collection("doctors");

//       const [doctorDocuments, total] = await Promise.all([
//         doctorsCollection
//           .find(filter)
//           .sort({
//             ratingAverage: -1,
//             ratingCount: -1,
//             createdAt: -1,
//             _id: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         doctorsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         doctors: doctorDocuments.map(getPublicDoctor),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get public doctors error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve public doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public single doctor details
// ========================================================= */

// app.get(
//   "/api/v1/doctors/:doctorId",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         doctor: getPublicDoctor(doctor),
//       });
//     } catch (error) {
//       console.error("Get public doctor details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctor reviews
// ========================================================= */

// app.get(
//   "/api/v1/doctors/:doctorId/reviews",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = getPositiveInteger(req.query.limit, 10, 50);

//       const reviewsCollection = database.collection("reviews");

//       const [reviewDocuments, total] = await Promise.all([
//         reviewsCollection
//           .find({
//             doctorId,
//           })
//           .sort({
//             updatedAt: -1,
//             createdAt: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         reviewsCollection.countDocuments({
//           doctorId,
//         }),
//       ]);

//       res.status(200).json({
//         success: true,
//         reviews: reviewDocuments.map(formatReview),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get doctor reviews error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor reviews",
//       });
//     }
//   },
// );

// /* =========================================================
//    Create doctor review
// ========================================================= */

// app.post(
//   "/api/v1/doctors/:doctorId/reviews",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot submit a rating or review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const rating = Math.floor(Number(req.body.rating));
//       const reviewText = getDoctorString(req.body.review);

//       if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
//         res.status(400).json({
//           success: false,
//           message: "Rating must be a number from 1 to 5",
//         });

//         return;
//       }

//       if (reviewText.length > 2000) {
//         res.status(400).json({
//           success: false,
//           message: "Review cannot contain more than 2000 characters",
//         });

//         return;
//       }

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(doctor.userId) === currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "A doctor cannot review their own profile",
//         });

//         return;
//       }

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         doctorId,
//         userId: currentUserId,
//       });

//       if (existingReview) {
//         res.status(409).json({
//           success: false,
//           message:
//             "You have already reviewed this doctor. Please edit your existing review.",
//           code: "REVIEW_ALREADY_EXISTS",
//         });

//         return;
//       }

//       const now = new Date();

//       const reviewDocument = {
//         doctorId,
//         doctorUserId: getDoctorString(doctor.userId),
//         userId: currentUserId,
//         userName: getDoctorString(currentUser.name),
//         userEmail: normalizeDoctorEmail(currentUser.email),
//         userImage: getDoctorString(currentUser.image) || null,
//         rating,
//         review: reviewText,
//         createdAt: now,
//         updatedAt: now,
//       };

//       const insertResult = await reviewsCollection.insertOne(reviewDocument);

//       await refreshDoctorRatingStats(doctorId);

//       const createdReview = await reviewsCollection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "Rating and review submitted successfully",
//         review: createdReview ? formatReview(createdReview) : null,
//       });
//     } catch (error) {
//       console.error("Create doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to submit rating and review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Update doctor review
// ========================================================= */

// app.patch(
//   "/api/v1/doctors/:doctorId/reviews/:reviewId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot edit a review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const reviewId = getDoctorString(req.params.reviewId);
//       const rating = Math.floor(Number(req.body.rating));
//       const reviewText = getDoctorString(req.body.review);

//       if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
//         res.status(400).json({
//           success: false,
//           message: "Rating must be a number from 1 to 5",
//         });

//         return;
//       }

//       if (reviewText.length > 2000) {
//         res.status(400).json({
//           success: false,
//           message: "Review cannot contain more than 2000 characters",
//         });

//         return;
//       }

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         $and: [
//           getDoctorFilter(reviewId),
//           {
//             doctorId,
//           },
//         ],
//       });

//       if (!existingReview) {
//         res.status(404).json({
//           success: false,
//           message: "Review was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(existingReview.userId) !== currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can edit only your own review",
//         });

//         return;
//       }

//       const updatedReview = await reviewsCollection.findOneAndUpdate(
//         {
//           _id: existingReview._id,
//         },
//         {
//           $set: {
//             rating,
//             review: reviewText,
//             updatedAt: new Date(),
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       await refreshDoctorRatingStats(doctorId);

//       res.status(200).json({
//         success: true,
//         message: "Rating and review updated successfully",
//         review: updatedReview ? formatReview(updatedReview) : null,
//       });
//     } catch (error) {
//       console.error("Update doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update rating and review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Delete doctor review
// ========================================================= */

// app.delete(
//   "/api/v1/doctors/:doctorId/reviews/:reviewId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot delete a review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const reviewId = getDoctorString(req.params.reviewId);

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         $and: [
//           getDoctorFilter(reviewId),
//           {
//             doctorId,
//           },
//         ],
//       });

//       if (!existingReview) {
//         res.status(404).json({
//           success: false,
//           message: "Review was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(existingReview.userId) !== currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can delete only your own review",
//         });

//         return;
//       }

//       await reviewsCollection.deleteOne({
//         _id: existingReview._id,
//       });

//       await refreshDoctorRatingStats(doctorId);

//       res.status(200).json({
//         success: true,
//         message: "Review deleted successfully",
//         deletedReviewId: reviewId,
//       });
//     } catch (error) {
//       console.error("Delete doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Appointment eligibility
// ========================================================= */

// app.get(
//   "/api/v1/appointments/eligibility/:doctorId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);

//       if (role !== "patient") {
//         res.status(403).json({
//           success: false,
//           canBook: false,
//           code: "PATIENT_ONLY",
//           message: "Only patients can take a doctor appointment.",
//         });

//         return;
//       }

//       if (status === "blocked") {
//         res.status(403).json({
//           success: false,
//           canBook: false,
//           code: "ACCOUNT_BLOCKED",
//           message:
//             "You are restricted by the administrator and cannot take an appointment.",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           canBook: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);

//       const existingAppointment = await database
//         .collection("appointments")
//         .findOne({
//           doctorId,
//           patientUserId,
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//       if (existingAppointment) {
//         res.status(200).json({
//           success: true,
//           canBook: false,
//           code: "APPOINTMENT_ALREADY_EXISTS",
//           message:
//             "You already have a pending or approved appointment with this doctor.",
//           appointment: formatAppointment(existingAppointment),
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         canBook: true,
//         message: "You can take an appointment with this doctor.",
//       });
//     } catch (error) {
//       console.error("Appointment eligibility error:", error);

//       res.status(500).json({
//         success: false,
//         canBook: false,
//         message: "Failed to check appointment eligibility",
//       });
//     }
//   },
// );

// /* =========================================================
//    Create appointment
// ========================================================= */

// app.post(
//   "/api/v1/appointments",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);

//       if (role !== "patient") {
//         res.status(403).json({
//           success: false,
//           message: "Only patients can take a doctor appointment.",
//           code: "PATIENT_ONLY",
//         });

//         return;
//       }

//       if (status === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot take an appointment.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.body.doctorId);
//       const patientName = getDoctorString(req.body.patientName);
//       const phone = getDoctorString(req.body.phone);
//       const address = getDoctorString(req.body.address);
//       const problemTitle = getDoctorString(req.body.problemTitle);
//       const symptomsDescription = getDoctorString(req.body.symptomsDescription);
//       const appointmentDate = getDoctorString(req.body.appointmentDate);
//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       if (
//         !doctorId ||
//         !patientName ||
//         !phone ||
//         !address ||
//         !problemTitle ||
//         !symptomsDescription ||
//         !appointmentDate ||
//         !appointmentTime
//       ) {
//         res.status(400).json({
//           success: false,
//           message:
//             "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
//         });

//         return;
//       }

//       if (symptomsDescription.length > 5000) {
//         res.status(400).json({
//           success: false,
//           message:
//             "Symptoms description cannot contain more than 5000 characters",
//         });

//         return;
//       }

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentsCollection = database.collection("appointments");

//       const existingAppointment = await appointmentsCollection.findOne({
//         doctorId,
//         patientUserId,
//         status: {
//           $in: ACTIVE_APPOINTMENT_STATUSES,
//         },
//       });

//       if (existingAppointment) {
//         res.status(409).json({
//           success: false,
//           message:
//             "You already have a pending or approved appointment with this doctor.",
//           code: "APPOINTMENT_ALREADY_EXISTS",
//           appointment: formatAppointment(existingAppointment),
//         });

//         return;
//       }

//       const now = new Date();

//       const appointmentDocument = {
//         doctorId,
//         doctorUserId: getDoctorString(doctor.userId),
//         doctorName: getDoctorString(doctor.name),
//         doctorImage: getDoctorString(doctor.image) || null,
//         specialization: getDoctorString(doctor.specialization),
//         hospital:
//           getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
//         patientUserId,
//         patientName,
//         patientEmail: normalizeDoctorEmail(currentUser.email),
//         patientImage: getDoctorString(currentUser.image) || null,
//         phone,
//         address,
//         problemTitle,
//         symptomsDescription,
//         appointmentDate,
//         appointmentTime,
//         status: "pending" as const,
//         createdAt: now,
//         updatedAt: now,
//       };

//       const insertResult =
//         await appointmentsCollection.insertOne(appointmentDocument);

//       const createdAppointment = await appointmentsCollection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "Appointment request submitted successfully",
//         appointment: createdAppointment
//           ? formatAppointment(createdAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Create appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to submit appointment request",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient appointment helpers
// ========================================================= */

// const getPatientAppointment = async (
//   patientUserId: string,
//   appointmentId: string,
// ): Promise<Document | null> => {
//   if (!database) {
//     return null;
//   }

//   return database.collection("appointments").findOne({
//     $and: [
//       getDoctorFilter(appointmentId),
//       {
//         patientUserId,
//       },
//     ],
//   });
// };

// const getAppointmentDoctor = async (
//   appointment: Document,
// ): Promise<Document | null> => {
//   if (!database) {
//     return null;
//   }

//   const doctorId = getDoctorString(appointment.doctorId);

//   if (!doctorId) {
//     return null;
//   }

//   return database.collection("doctors").findOne(getDoctorFilter(doctorId));
// };

// const validatePatientAppointmentInput = (
//   body: Record<string, unknown>,
// ):
//   | {
//       success: true;
//       values: {
//         patientName: string;
//         phone: string;
//         address: string;
//         problemTitle: string;
//         symptomsDescription: string;
//         appointmentDate: string;
//         appointmentTime: string;
//       };
//     }
//   | {
//       success: false;
//       message: string;
//     } => {
//   const patientName = getDoctorString(body.patientName);
//   const phone = getDoctorString(body.phone);
//   const address = getDoctorString(body.address);
//   const problemTitle = getDoctorString(body.problemTitle);
//   const symptomsDescription = getDoctorString(body.symptomsDescription);
//   const appointmentDate = getDoctorString(body.appointmentDate);
//   const appointmentTime = getDoctorString(body.appointmentTime);

//   if (
//     !patientName ||
//     !phone ||
//     !address ||
//     !problemTitle ||
//     !symptomsDescription ||
//     !appointmentDate ||
//     !appointmentTime
//   ) {
//     return {
//       success: false,
//       message:
//         "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
//     };
//   }

//   if (patientName.length > 150) {
//     return {
//       success: false,
//       message: "Patient name cannot contain more than 150 characters",
//     };
//   }

//   if (phone.length > 40) {
//     return {
//       success: false,
//       message: "Phone number cannot contain more than 40 characters",
//     };
//   }

//   if (address.length > 500) {
//     return {
//       success: false,
//       message: "Address cannot contain more than 500 characters",
//     };
//   }

//   if (problemTitle.length > 250) {
//     return {
//       success: false,
//       message: "Health problem title cannot contain more than 250 characters",
//     };
//   }

//   if (symptomsDescription.length > 5000) {
//     return {
//       success: false,
//       message: "Symptoms description cannot contain more than 5000 characters",
//     };
//   }

//   const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//   const timePattern = /^\d{2}:\d{2}$/;

//   if (
//     !datePattern.test(appointmentDate) ||
//     !timePattern.test(appointmentTime)
//   ) {
//     return {
//       success: false,
//       message: "A valid appointment date and time are required",
//     };
//   }

//   const today = new Date().toISOString().slice(0, 10);

//   if (appointmentDate < today) {
//     return {
//       success: false,
//       message: "Appointment date cannot be in the past",
//     };
//   }

//   return {
//     success: true,
//     values: {
//       patientName,
//       phone,
//       address,
//       problemTitle,
//       symptomsDescription,
//       appointmentDate,
//       appointmentTime,
//     },
//   };
// };

// /* =========================================================
//    Patient appointments list
// ========================================================= */

// app.get(
//   "/api/v1/patient/appointments",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;
//       const appointmentsCollection = database.collection("appointments");
//       const filter: Filter<Document> = { patientUserId };

//       const [appointmentDocuments, total] = await Promise.all([
//         appointmentsCollection
//           .find(filter)
//           .sort({ createdAt: -1, _id: -1 })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),
//         appointmentsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         appointments: appointmentDocuments.map(formatAppointment),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get patient appointments error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient single appointment details
// ========================================================= */

// app.get(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);

//       if (!appointmentId) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment ID is required",
//         });
//         return;
//       }

//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       const doctor = await getAppointmentDoctor(appointment);

//       res.status(200).json({
//         success: true,
//         appointment: formatAppointment(appointment),
//         doctor: doctor ? getPublicDoctor(doctor) : null,
//       });
//     } catch (error) {
//       console.error("Get patient appointment details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointment details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient edit appointment
// ========================================================= */

// app.patch(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus !== "pending" && currentStatus !== "rejected") {
//         res.status(409).json({
//           success: false,
//           message: "Only pending or rejected appointments can be edited",
//         });
//         return;
//       }

//       const validation = validatePatientAppointmentInput(
//         req.body as Record<string, unknown>,
//       );

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       if (currentStatus === "rejected") {
//         const anotherActiveAppointment = await database
//           .collection("appointments")
//           .findOne({
//             _id: { $ne: appointment._id },
//             doctorId: getDoctorString(appointment.doctorId),
//             patientUserId,
//             status: { $in: ACTIVE_APPOINTMENT_STATUSES },
//           });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "You already have another pending or approved appointment with this doctor.",
//           });
//           return;
//         }
//       }

//       const now = new Date();
//       const updatedAppointment = await database
//         .collection("appointments")
//         .findOneAndUpdate(
//           { _id: appointment._id },
//           {
//             $set: {
//               ...validation.values,
//               status: "pending",
//               rejectionReason: null,
//               rejectedAt: null,
//               approvedAt: null,
//               completedAt: null,
//               updatedAt: now,
//             },
//           },
//           { returnDocument: "after" },
//         );

//       res.status(200).json({
//         success: true,
//         message:
//           currentStatus === "rejected"
//             ? "Appointment updated and resubmitted successfully"
//             : "Appointment updated successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update patient appointment error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient cancel and delete appointment
// ========================================================= */

// app.delete(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       if (getDoctorString(appointment.status) === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be cancelled or deleted",
//         });
//         return;
//       }

//       const deleteResult = await database
//         .collection("appointments")
//         .deleteOne({ _id: appointment._id });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Appointment could not be cancelled",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Appointment cancelled and removed successfully",
//         deletedAppointmentId: appointmentId,
//       });
//     } catch (error) {
//       console.error("Cancel patient appointment error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to cancel appointment",
//       });
//     }
//   },
// );

// const getAppointmentListFilter = (
//   req: AuthenticatedRequest,
// ): Filter<Document> => {
//   const conditions: Filter<Document>[] = [];
//   const status = getDoctorString(req.query.status);
//   const search = getDoctorString(req.query.search);

//   if (
//     status === "pending" ||
//     status === "approved" ||
//     status === "completed" ||
//     status === "rejected"
//   ) {
//     conditions.push({
//       status,
//     });
//   }

//   if (search) {
//     const safeSearch = escapeDoctorSearch(search);

//     conditions.push({
//       $or: [
//         {
//           patientName: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           patientEmail: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           doctorName: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           problemTitle: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//       ],
//     });
//   }

//   return conditions.length
//     ? {
//         $and: conditions,
//       }
//     : {};
// };

// const sendAppointmentList = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   additionalFilter: Filter<Document> = {},
// ): Promise<void> => {
//   if (!database) {
//     res.status(503).json({
//       success: false,
//       message: "Database is not connected",
//     });

//     return;
//   }

//   const page = getPositiveInteger(req.query.page, 1, 100000);
//   const limit = 10;

//   const queryFilter = getAppointmentListFilter(req);

//   const filter: Filter<Document> = {
//     $and: [queryFilter, additionalFilter],
//   };

//   const appointmentsCollection = database.collection("appointments");

//   const [appointmentDocuments, total] = await Promise.all([
//     appointmentsCollection
//       .find(filter)
//       .sort({
//         createdAt: -1,
//         _id: -1,
//       })
//       .skip((page - 1) * limit)
//       .limit(limit)
//       .toArray(),

//     appointmentsCollection.countDocuments(filter),
//   ]);

//   const appointmentsWithImages =
//     await attachPatientImages(appointmentDocuments);

//   res.status(200).json({
//     success: true,
//     appointments: appointmentsWithImages.map(formatAppointment),
//     pagination: {
//       page,
//       limit,
//       total,
//       totalPages: Math.max(1, Math.ceil(total / limit)),
//     },
//   });
// };

// /* =========================================================
//    Admin appointment management
// ========================================================= */

// app.get(
//   "/api/v1/admin/appointments",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       await sendAppointmentList(req, res);
//     } catch (error) {
//       console.error("Get admin appointments error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Doctor appointment management
// ========================================================= */

// app.get(
//   "/api/v1/doctor/appointments",
//   verifyToken,
//   verifyDoctor,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       await sendAppointmentList(req, res, {
//         doctorUserId,
//       });
//     } catch (error) {
//       console.error("Get doctor appointments error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Doctor single appointment details
// ========================================================= */

// app.get(
//   "/api/v1/doctor/appointments/:appointmentId",
//   verifyToken,
//   verifyDoctor,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       if (!appointmentId) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment ID is required",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       const appointment = await database.collection("appointments").findOne({
//         $and: [
//           getDoctorFilter(appointmentId),
//           {
//             doctorUserId,
//           },
//         ],
//       });

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const [appointmentWithImage] = await attachPatientImages([appointment]);

//       res.status(200).json({
//         success: true,
//         appointment: formatAppointment(appointmentWithImage),
//       });
//     } catch (error) {
//       console.error("Get doctor appointment details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointment details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// /* =========================================================
//    Doctor appointment reschedule
// ========================================================= */

// app.patch(
//   "/api/v1/doctor/appointments/:appointmentId/reschedule",
//   verifyToken,
//   verifyDoctor,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentDate = getDoctorString(req.body.appointmentDate);

//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       const rescheduleReason = getDoctorString(req.body.rescheduleReason);

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       if (rescheduleReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Reschedule reason cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(appointment.doctorUserId) !== doctorUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can reschedule only your own appointments",
//         });

//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (
//         currentStatus !== "pending" &&
//         currentStatus !== "approved" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message:
//             "Only pending, approved or rejected appointments can be rescheduled",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: {
//             appointmentDate,
//             appointmentTime,
//             rescheduleReason: rescheduleReason || null,
//             rescheduledAt: now,
//             rescheduledBy: doctorUserId,
//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message: "Appointment rescheduled successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Doctor reschedule appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to reschedule appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin-only appointment reschedule
// ========================================================= */

// app.patch(
//   "/api/v1/admin/appointments/:appointmentId/reschedule",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentDate = getDoctorString(req.body.appointmentDate);

//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       const rescheduleReason = getDoctorString(req.body.rescheduleReason);

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       if (rescheduleReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Reschedule reason cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed" || currentStatus === "rejected") {
//         res.status(409).json({
//           success: false,
//           message: "A completed or rejected appointment cannot be rescheduled",
//         });

//         return;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: {
//             appointmentDate,
//             appointmentTime,
//             rescheduleReason: rescheduleReason || null,
//             rescheduledAt: new Date(),
//             rescheduledBy: req.userId || null,
//             updatedAt: new Date(),
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message: "Appointment rescheduled successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Reschedule appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to reschedule appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// app.patch(
//   "/api/v1/appointments/:appointmentId/status",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const userStatus = getNormalizedUserStatus(currentUser);

//       if (role !== "admin" && role !== "doctor") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Only an administrator or doctor can update appointment status.",
//         });

//         return;
//       }

//       if (userStatus === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Your account is blocked. You can view appointments but cannot update them.",
//           code: "READ_ONLY_ACCOUNT",
//         });

//         return;
//       }

//       const requestedStatus = getDoctorString(req.body.status);
//       const rejectionReason = getDoctorString(req.body.rejectionReason);

//       if (
//         requestedStatus !== "approved" &&
//         requestedStatus !== "completed" &&
//         requestedStatus !== "rejected"
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "Status must be approved, completed or rejected",
//         });

//         return;
//       }

//       if (requestedStatus === "rejected" && !rejectionReason) {
//         res.status(400).json({
//           success: false,
//           message:
//             "A rejection message is required when rejecting an appointment",
//         });

//         return;
//       }

//       if (rejectionReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Rejection message cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       if (role === "doctor") {
//         const currentUserId = getDoctorDocumentId(currentUser);

//         if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
//           res.status(403).json({
//             success: false,
//             message: "You can update only your own appointments",
//           });

//           return;
//         }
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be changed",
//         });

//         return;
//       }

//       if (requestedStatus === "completed" && currentStatus !== "approved") {
//         res.status(409).json({
//           success: false,
//           message: "Only an approved appointment can be marked as completed",
//         });

//         return;
//       }

//       if (
//         requestedStatus === "approved" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or rejected appointment can be approved",
//         });

//         return;
//       }

//       if (
//         requestedStatus === "rejected" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "approved"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or approved appointment can be rejected",
//         });

//         return;
//       }

//       if (requestedStatus === "approved" && currentStatus === "rejected") {
//         const anotherActiveAppointment = await appointmentsCollection.findOne({
//           _id: {
//             $ne: appointment._id,
//           },
//           doctorId: getDoctorString(appointment.doctorId),
//           patientUserId: getDoctorString(appointment.patientUserId),
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "This patient already has another pending or approved appointment with you.",
//           });

//           return;
//         }
//       }

//       const now = new Date();

//       const statusFields: Record<string, unknown> = {
//         status: requestedStatus as AppointmentStatus,
//         rejectionReason:
//           requestedStatus === "rejected" ? rejectionReason : null,
//         updatedAt: now,
//       };

//       if (requestedStatus === "approved") {
//         statusFields.approvedAt = now;
//         statusFields.rejectedAt = null;
//         statusFields.rejectionReason = null;
//       }

//       if (requestedStatus === "completed") {
//         statusFields.completedAt = now;
//       }

//       if (requestedStatus === "rejected") {
//         statusFields.rejectedAt = now;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: statusFields,
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message:
//           requestedStatus === "approved"
//             ? "Appointment approved successfully"
//             : requestedStatus === "completed"
//               ? "Consultation completed successfully."
//               : "Appointment rejected successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update appointment status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment status",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// app.patch(
//   "/api/v1/appointments/:appointmentId/status",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const userStatus = getNormalizedUserStatus(currentUser);

//       if (role !== "admin" && role !== "doctor") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Only an administrator or doctor can update appointment status.",
//         });
//         return;
//       }

//       if (userStatus === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Your account is blocked. You can view appointments but cannot update them.",
//           code: "READ_ONLY_ACCOUNT",
//         });
//         return;
//       }

//       const requestedStatus = getDoctorString(req.body.status);
//       const rejectionReason = getDoctorString(req.body.rejectionReason);

//       if (
//         requestedStatus !== "approved" &&
//         requestedStatus !== "completed" &&
//         requestedStatus !== "rejected"
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "Status must be approved, completed or rejected",
//         });
//         return;
//       }

//       if (requestedStatus === "rejected" && !rejectionReason) {
//         res.status(400).json({
//           success: false,
//           message:
//             "A rejection message is required when rejecting an appointment",
//         });
//         return;
//       }

//       if (rejectionReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Rejection message cannot contain more than 1000 characters",
//         });
//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointmentsCollection = database.collection("appointments");
//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       if (role === "doctor") {
//         const currentUserId = getDoctorDocumentId(currentUser);
//         if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
//           res.status(403).json({
//             success: false,
//             message: "You can update only your own appointments",
//           });
//           return;
//         }
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be changed",
//         });
//         return;
//       }

//       if (requestedStatus === "completed" && currentStatus !== "approved") {
//         res.status(409).json({
//           success: false,
//           message: "Only an approved appointment can be marked as completed",
//         });
//         return;
//       }

//       if (
//         requestedStatus === "approved" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or rejected appointment can be approved",
//         });
//         return;
//       }

//       if (
//         requestedStatus === "rejected" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "approved"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or approved appointment can be rejected",
//         });
//         return;
//       }

//       if (requestedStatus === "approved" && currentStatus === "rejected") {
//         const anotherActiveAppointment = await appointmentsCollection.findOne({
//           _id: {
//             $ne: appointment._id,
//           },
//           doctorId: getDoctorString(appointment.doctorId),
//           patientUserId: getDoctorString(appointment.patientUserId),
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "This patient already has another pending or approved appointment with you.",
//           });
//           return;
//         }
//       }

//       const now = new Date();

//       const statusFields: Record<string, unknown> = {
//         status: requestedStatus as AppointmentStatus,
//         rejectionReason:
//           requestedStatus === "rejected" ? rejectionReason : null,
//         updatedAt: now,
//       };

//       if (requestedStatus === "approved") {
//         statusFields.approvedAt = now;
//         statusFields.rejectedAt = null;
//         statusFields.rejectionReason = null;
//       }

//       if (requestedStatus === "completed") {
//         statusFields.completedAt = now;
//       }

//       if (requestedStatus === "rejected") {
//         statusFields.rejectedAt = now;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: statusFields,
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message:
//           requestedStatus === "approved"
//             ? "Appointment approved successfully"
//             : requestedStatus === "completed"
//               ? "Consultation completed successfully."
//               : "Appointment rejected successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update appointment status error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment status",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin Dashboard Statistics                               <-- ADD THIS HERE
// ========================================================= */

// app.get(
//   "/api/v1/admin/dashboard/stats",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     console.log("✅ Admin dashboard stats route called!"); // Debug log
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       // Get total patients
//       const totalPatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient" });

//       // Get active patients
//       const activePatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient", status: "active" });

//       // Get blocked patients
//       const blockedPatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient", status: "blocked" });

//       // Get total doctors
//       const totalDoctors = await database
//         .collection("doctors")
//         .countDocuments();

//       // Get active doctors
//       const activeDoctors = await database
//         .collection("doctors")
//         .countDocuments({ status: "active" });

//       // Get blocked doctors
//       const blockedDoctors = await database
//         .collection("doctors")
//         .countDocuments({ status: "blocked" });

//       // Get appointment counts by status
//       const appointmentCounts = await database
//         .collection("appointments")
//         .aggregate([
//           {
//             $group: {
//               _id: "$status",
//               count: { $sum: 1 },
//             },
//           },
//         ])
//         .toArray();

//       // Create status count object with default values
//       const statusCounts: Record<string, number> = {
//         pending: 0,
//         approved: 0,
//         completed: 0,
//         rejected: 0,
//       };

//       appointmentCounts.forEach((item) => {
//         const status = item._id || "pending";
//         if (status in statusCounts) {
//           statusCounts[status] = item.count;
//         }
//       });

//       // Get total appointments
//       const totalAppointments = await database
//         .collection("appointments")
//         .countDocuments();

//       // Get completed consultations
//       const completedConsultations = statusCounts.completed;

//       // Get monthly appointment trends (last 6 months)
//       const sixMonthsAgo = new Date();
//       sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

//       const monthlyTrends = await database
//         .collection("appointments")
//         .aggregate([
//           {
//             $match: {
//               createdAt: { $gte: sixMonthsAgo },
//             },
//           },
//           {
//             $group: {
//               _id: {
//                 year: { $year: "$createdAt" },
//                 month: { $month: "$createdAt" },
//               },
//               count: { $sum: 1 },
//               pending: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
//                 },
//               },
//               approved: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
//                 },
//               },
//               completed: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
//                 },
//               },
//               rejected: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
//                 },
//               },
//             },
//           },
//           {
//             $sort: { "_id.year": 1, "_id.month": 1 },
//           },
//         ])
//         .toArray();

//       // Get appointment status breakdown for charts
//       const statusColors: Record<string, string> = {
//         pending: "#FBBF24",
//         approved: "#60A5FA",
//         completed: "#34D399",
//         rejected: "#F87171",
//       };

//       const statusData = appointmentCounts.map((item) => ({
//         name: item._id || "unknown",
//         value: item.count,
//         fill: statusColors[item._id] || "#9CA3AF",
//       }));

//       // Format monthly data for charts
//       const monthNames = [
//         "Jan",
//         "Feb",
//         "Mar",
//         "Apr",
//         "May",
//         "Jun",
//         "Jul",
//         "Aug",
//         "Sep",
//         "Oct",
//         "Nov",
//         "Dec",
//       ];
//       const monthlyData = monthlyTrends.map((item) => ({
//         month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
//         pending: item.pending || 0,
//         approved: item.approved || 0,
//         completed: item.completed || 0,
//         rejected: item.rejected || 0,
//         total: item.count || 0,
//       }));

//       // Get recent appointments (last 10)
//       const recentAppointments = await database
//         .collection("appointments")
//         .find()
//         .sort({ createdAt: -1 })
//         .limit(10)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         data: {
//           overview: {
//             totalPatients,
//             activePatients,
//             blockedPatients,
//             totalDoctors,
//             activeDoctors,
//             blockedDoctors,
//             totalAppointments,
//             completedConsultations,
//             appointmentStatus: statusCounts,
//           },
//           charts: {
//             appointmentStatus: statusData,
//             monthlyTrends: monthlyData,
//           },
//           recentAppointments: recentAppointments.map((app) => ({
//             id: app._id,
//             patientName: app.patientName,
//             doctorName: app.doctorName,
//             specialization: app.specialization,
//             appointmentDate: app.appointmentDate,
//             appointmentTime: app.appointmentTime,
//             status: app.status,
//             createdAt: app.createdAt,
//           })),
//         },
//       });
//     } catch (error) {
//       console.error("Dashboard stats error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to fetch dashboard statistics",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin Dashboard Statistics                         <-- ADD THIS HERE
// ========================================================= */

// // app.get(
// //   "/api/v1/admin/dashboard/stats",
// //   verifyToken,
// //   verifyAdmin,
// //   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
// //     console.log("✅ Admin dashboard stats route called!"); // Debug log
// //     try {
// //       if (!database) {
// //         res.status(503).json({
// //           success: false,
// //           message: "Database is not connected",
// //         });
// //         return;
// //       }

// //       // Get total patients
// //       const totalPatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient" });

// //       // Get active patients
// //       const activePatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient", status: "active" });

// //       // Get blocked patients
// //       const blockedPatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient", status: "blocked" });

// //       // Get total doctors
// //       const totalDoctors = await database
// //         .collection("doctors")
// //         .countDocuments();

// //       // Get active doctors
// //       const activeDoctors = await database
// //         .collection("doctors")
// //         .countDocuments({ status: "active" });

// //       // Get blocked doctors
// //       const blockedDoctors = await database
// //         .collection("doctors")
// //         .countDocuments({ status: "blocked" });

// //       // Get appointment counts by status
// //       const appointmentCounts = await database
// //         .collection("appointments")
// //         .aggregate([
// //           {
// //             $group: {
// //               _id: "$status",
// //               count: { $sum: 1 },
// //             },
// //           },
// //         ])
// //         .toArray();

// //       // Create status count object with default values
// //       const statusCounts: Record<string, number> = {
// //         pending: 0,
// //         approved: 0,
// //         completed: 0,
// //         rejected: 0,
// //       };

// //       appointmentCounts.forEach((item) => {
// //         const status = item._id || "pending";
// //         if (status in statusCounts) {
// //           statusCounts[status] = item.count;
// //         }
// //       });

// //       // Get total appointments
// //       const totalAppointments = await database
// //         .collection("appointments")
// //         .countDocuments();

// //       // Get completed consultations
// //       const completedConsultations = statusCounts.completed;

// //       // Get monthly appointment trends (last 6 months)
// //       const sixMonthsAgo = new Date();
// //       sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

// //       const monthlyTrends = await database
// //         .collection("appointments")
// //         .aggregate([
// //           {
// //             $match: {
// //               createdAt: { $gte: sixMonthsAgo },
// //             },
// //           },
// //           {
// //             $group: {
// //               _id: {
// //                 year: { $year: "$createdAt" },
// //                 month: { $month: "$createdAt" },
// //               },
// //               count: { $sum: 1 },
// //               pending: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
// //                 },
// //               },
// //               approved: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
// //                 },
// //               },
// //               completed: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
// //                 },
// //               },
// //               rejected: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
// //                 },
// //               },
// //             },
// //           },
// //           {
// //             $sort: { "_id.year": 1, "_id.month": 1 },
// //           },
// //         ])
// //         .toArray();

// //       // Get appointment status breakdown for charts
// //       const statusColors: Record<string, string> = {
// //         pending: "#FBBF24",
// //         approved: "#60A5FA",
// //         completed: "#34D399",
// //         rejected: "#F87171",
// //       };

// //       const statusData = appointmentCounts.map((item) => ({
// //         name: item._id || "unknown",
// //         value: item.count,
// //         fill: statusColors[item._id] || "#9CA3AF",
// //       }));

// //       // Format monthly data for charts
// //       const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// //       const monthlyData = monthlyTrends.map((item) => ({
// //         month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
// //         pending: item.pending || 0,
// //         approved: item.approved || 0,
// //         completed: item.completed || 0,
// //         rejected: item.rejected || 0,
// //         total: item.count || 0,
// //       }));

// //       // Get recent appointments (last 10)
// //       const recentAppointments = await database
// //         .collection("appointments")
// //         .find()
// //         .sort({ createdAt: -1 })
// //         .limit(10)
// //         .toArray();

// //       res.status(200).json({
// //         success: true,
// //         data: {
// //           overview: {
// //             totalPatients,
// //             activePatients,
// //             blockedPatients,
// //             totalDoctors,
// //             activeDoctors,
// //             blockedDoctors,
// //             totalAppointments,
// //             completedConsultations,
// //             appointmentStatus: statusCounts,
// //           },
// //           charts: {
// //             appointmentStatus: statusData,
// //             monthlyTrends: monthlyData,
// //           },
// //           recentAppointments: recentAppointments.map((app) => ({
// //             id: app._id,
// //             patientName: app.patientName,
// //             doctorName: app.doctorName,
// //             specialization: app.specialization,
// //             appointmentDate: app.appointmentDate,
// //             appointmentTime: app.appointmentTime,
// //             status: app.status,
// //             createdAt: app.createdAt,
// //           })),
// //         },
// //       });
// //     } catch (error) {
// //       console.error("Dashboard stats error:", error);
// //       res.status(500).json({
// //         success: false,
// //         message: "Failed to fetch dashboard statistics",
// //       });
// //     }
// //   },
// // );

// /* =========================================================
//    SebaSathi AI Health Assistant (Groq)
// ========================================================= */

// type AIHealthMessageRole = "user" | "assistant";

// type AIHealthUrgency = "routine" | "soon" | "urgent" | "emergency";

// type AIHealthStreamStage =
//   | "thinking"
//   | "tool"
//   | "answering"
//   | "structuring"
//   | "saving";

// interface AIHealthMessage {
//   role: AIHealthMessageRole;
//   content: string;
// }

// interface AIHealthNavigationRoute {
//   label: string;
//   href: string;
//   description: string;
// }

// interface AIHealthNavigationAction {
//   label: string;
//   href: string;
//   reason: string;
// }

// interface AIHealthAssistantResponse {
//   reply: string;
//   urgencyLevel: AIHealthUrgency;
//   suggestedSpecialists: string[];
//   recommendedActions: string[];
//   warningSigns: string[];
//   followUpQuestions: string[];
//   suggestedPrompts: string[];
//   navigationActions: AIHealthNavigationAction[];
//   decisionBasis: string;
//   toolsUsed: string[];
//   contextMemoryUsed: boolean;
//   disclaimer: string;
// }

// interface AIHealthStoredMessage extends AIHealthMessage {
//   id: string;
//   assistant?: AIHealthAssistantResponse;
//   createdAt: Date;
// }

// interface AIHealthSummaryReport {
//   reportTitle: string;
//   conciseSummary: string;
//   chiefConcerns: string[];
//   symptoms: string[];
//   durationAndPattern: string;
//   severity: string;
//   urgencyLevel: AIHealthUrgency;
//   redFlags: string[];
//   suggestedSpecialists: string[];
//   selfCareGuidance: string[];
//   questionsForDoctor: string[];
//   emergencyAdvice: string;
//   disclaimer: string;
// }

// interface AIHealthConversationDocument {
//   _id?: ObjectId;
//   title?: string;
//   messages: AIHealthStoredMessage[];
//   summaryHistoryId?: string | null;
//   summaryReport?: AIHealthSummaryReport | null;
//   updatedAt?: Date;
//   lastMessageAt?: Date;
// }

// interface AIHealthApplicationContext {
//   user: {
//     id: string;
//     name: string;
//     role: UserRole;
//   };
//   routes: AIHealthNavigationRoute[];
//   doctorDirectory: {
//     activeDoctorCount: number;
//     specializations: string[];
//     highlightedDoctors: Array<{
//       id: string;
//       name: string;
//       specialization: string;
//       hospital: string;
//       ratingAverage: number;
//     }>;
//   } | null;
//   appointmentContext: {
//     total: number;
//     counts: Record<string, number>;
//     recentAppointments: Array<{
//       id: string;
//       doctorName: string;
//       patientName: string;
//       specialization: string;
//       appointmentDate: string;
//       appointmentTime: string;
//       status: string;
//     }>;
//   } | null;
//   recentHealthHistory: Array<{
//     id: string;
//     title: string;
//     urgencyLevel: AIHealthUrgency;
//     updatedAt: string | null;
//   }>;
//   toolsUsed: string[];
//   contextMemoryUsed: boolean;
// }

// const aiHealthRateLimit = new Map<
//   string,
//   {
//     startedAt: number;
//     count: number;
//   }
// >();

// const verifyAIHealthRateLimit = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): void => {
//   const key = req.userId || req.ip || "anonymous";
//   const now = Date.now();
//   const windowLength = 10 * 60 * 1000;
//   const maximumRequests = 40;
//   const current = aiHealthRateLimit.get(key);

//   if (!current || now - current.startedAt >= windowLength) {
//     aiHealthRateLimit.set(key, {
//       startedAt: now,
//       count: 1,
//     });

//     next();
//     return;
//   }

//   if (current.count >= maximumRequests) {
//     res.status(429).json({
//       success: false,
//       message:
//         "You have sent too many AI requests. Please try again after a few minutes.",
//       code: "AI_RATE_LIMITED",
//     });

//     return;
//   }

//   current.count += 1;
//   aiHealthRateLimit.set(key, current);
//   next();
// };

// const createAIHealthMessageId = (): string => {
//   return new ObjectId().toHexString();
// };

// const getAIHealthArray = (value: unknown, maximumItems = 8): string[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   return value
//     .map((item) => getDoctorString(item))
//     .filter(Boolean)
//     .slice(0, maximumItems);
// };

// const getAIHealthUrgency = (value: unknown): AIHealthUrgency => {
//   return value === "soon" || value === "urgent" || value === "emergency"
//     ? value
//     : "routine";
// };

// const getAIHealthBoolean = (value: unknown): boolean => value === true;

// const extractAIHealthJson = (content: string): Record<string, unknown> => {
//   const trimmed = content.trim();
//   const withoutFence = trimmed
//     .replace(/^```(?:json)?\s*/i, "")
//     .replace(/\s*```$/i, "")
//     .trim();

//   try {
//     const parsed = JSON.parse(withoutFence) as unknown;

//     if (typeof parsed === "object" && parsed !== null) {
//       return parsed as Record<string, unknown>;
//     }
//   } catch {
//     const firstBrace = withoutFence.indexOf("{");
//     const lastBrace = withoutFence.lastIndexOf("}");

//     if (firstBrace >= 0 && lastBrace > firstBrace) {
//       const parsed = JSON.parse(
//         withoutFence.slice(firstBrace, lastBrace + 1),
//       ) as unknown;

//       if (typeof parsed === "object" && parsed !== null) {
//         return parsed as Record<string, unknown>;
//       }
//     }
//   }

//   throw new Error("Groq returned an invalid structured response");
// };

// const normalizeAIHealthMessages = (
//   value: unknown,
//   options: {
//     requireLatestUser: boolean;
//     maximumMessages?: number;
//     maximumCharacters?: number;
//   },
// ):
//   | {
//       success: true;
//       messages: AIHealthMessage[];
//     }
//   | {
//       success: false;
//       message: string;
//     } => {
//   if (!Array.isArray(value)) {
//     return {
//       success: false,
//       message: "A conversation message list is required",
//     };
//   }

//   const maximumMessages = options.maximumMessages ?? 30;
//   const maximumCharacters = options.maximumCharacters ?? 30000;
//   const messages: AIHealthMessage[] = [];
//   let totalCharacters = 0;

//   for (const rawMessage of value.slice(-maximumMessages)) {
//     if (typeof rawMessage !== "object" || rawMessage === null) {
//       continue;
//     }

//     const message = rawMessage as Record<string, unknown>;
//     const role = message.role;
//     const content = getDoctorString(message.content);

//     if ((role !== "user" && role !== "assistant") || !content) {
//       continue;
//     }

//     if (content.length > 4000) {
//       return {
//         success: false,
//         message: "Each chat message cannot contain more than 4000 characters",
//       };
//     }

//     totalCharacters += content.length;

//     if (totalCharacters > maximumCharacters) {
//       return {
//         success: false,
//         message:
//           "This conversation is too long. Please generate a summary and start a new conversation.",
//       };
//     }

//     messages.push({
//       role,
//       content,
//     });
//   }

//   if (messages.length === 0) {
//     return {
//       success: false,
//       message: "At least one valid chat message is required",
//     };
//   }

//   if (!messages.some((message) => message.role === "user")) {
//     return {
//       success: false,
//       message: "At least one user message is required",
//     };
//   }

//   if (
//     options.requireLatestUser &&
//     messages[messages.length - 1]?.role !== "user"
//   ) {
//     return {
//       success: false,
//       message: "The latest conversation message must be from the user",
//     };
//   }

//   return {
//     success: true,
//     messages,
//   };
// };

// const PUBLIC_AI_HEALTH_NAVIGATION_ROUTES: AIHealthNavigationRoute[] = [
//   {
//     label: "Home",
//     href: "/",
//     description: "Open the SebaSathi home page.",
//   },
//   {
//     label: "Find Doctors",
//     href: "/find-doctors",
//     description: "Find active doctors and filter by specialization.",
//   },
//   {
//     label: "AI Health Assistant",
//     href: "/ai-health-assistant",
//     description: "Continue using the SebaSathi AI Health Assistant.",
//   },
//   {
//     label: "About Us",
//     href: "/about",
//     description: "Learn more about SebaSathi and its healthcare services.",
//   },
//   {
//     label: "Contact",
//     href: "/contact",
//     description: "Open the SebaSathi contact page.",
//   },
// ];

// const ROLE_AI_HEALTH_NAVIGATION_ROUTES: Record<
//   UserRole,
//   AIHealthNavigationRoute[]
// > = {
//   patient: [
//     {
//       label: "Patient Overview",
//       href: "/dashboard/patient",
//       description: "Open the patient's dashboard overview.",
//     },
//     {
//       label: "My Appointments",
//       href: "/dashboard/patient/appointments",
//       description: "View the patient's appointment requests and statuses.",
//     },
//     {
//       label: "Prescriptions",
//       href: "/dashboard/patient/prescriptions",
//       description: "View the patient's saved prescriptions.",
//     },
//     {
//       label: "Consultations",
//       href: "/dashboard/patient/consultations",
//       description: "View the patient's consultation records.",
//     },
//     {
//       label: "AI Health History",
//       href: "/dashboard/patient/ai-health-history",
//       description: "Review saved AI-generated health summaries.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/patient/my-profile",
//       description: "Open the patient's profile settings.",
//     },
//   ],
//   doctor: [
//     {
//       label: "Doctor Overview",
//       href: "/dashboard/doctor",
//       description: "Open the doctor's dashboard overview.",
//     },
//     {
//       label: "Appointments",
//       href: "/dashboard/doctor/patients-appointments",
//       description: "View appointments assigned to the signed-in doctor.",
//     },
//     {
//       label: "My Patients",
//       href: "/dashboard/doctor/patients",
//       description: "View the doctor's patient list.",
//     },
//     {
//       label: "Prescriptions",
//       href: "/dashboard/doctor/prescriptions",
//       description: "Create or review doctor prescription records.",
//     },
//     {
//       label: "Consultation Records",
//       href: "/dashboard/doctor/consultations",
//       description: "View the doctor's consultation records.",
//     },
//     {
//       label: "Availability",
//       href: "/dashboard/doctor/availability",
//       description: "Manage the doctor's availability schedule.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/doctor/my-profile",
//       description: "Open the doctor's profile settings.",
//     },
//   ],
//   admin: [
//     {
//       label: "Admin Overview",
//       href: "/dashboard/admin",
//       description: "Open the administrator dashboard overview.",
//     },
//     {
//       label: "Manage Users",
//       href: "/dashboard/admin/users",
//       description: "Open administrator user management.",
//     },
//     {
//       label: "Manage Doctors",
//       href: "/dashboard/admin/doctors",
//       description: "Open administrator doctor management.",
//     },
//     {
//       label: "Manage Appointments",
//       href: "/dashboard/admin/appointments",
//       description: "Open administrator appointment management.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/admin/my-profile",
//       description: "Open the administrator's profile settings.",
//     },
//   ],
// };

// const AI_HEALTH_NAVIGATION_ROUTE_ALIASES: Record<string, string> = {
//   "/doctors": "/find-doctors",
//   "/dashboard/doctor/appointments": "/dashboard/doctor/patients-appointments",
// };

// const normalizeAIHealthNavigationHref = (href: string): string => {
//   return AI_HEALTH_NAVIGATION_ROUTE_ALIASES[href] || href;
// };

// const getAllAIHealthNavigationRoutes = (): AIHealthNavigationRoute[] => [
//   ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.patient,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.doctor,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.admin,
// ];

// const getAIHealthNavigationRoutes = (
//   role: UserRole,
// ): AIHealthNavigationRoute[] => [
//   ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES[role],
// ];

// const getAIHealthNavigationActions = (
//   value: unknown,
//   allowedRoutes: AIHealthNavigationRoute[],
// ): AIHealthNavigationAction[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   const allowedByHref = new Map(
//     allowedRoutes.map((route) => [route.href, route] as const),
//   );

//   const actions: AIHealthNavigationAction[] = [];

//   for (const rawAction of value) {
//     if (typeof rawAction !== "object" || rawAction === null) {
//       continue;
//     }

//     const action = rawAction as Record<string, unknown>;
//     const href = normalizeAIHealthNavigationHref(getDoctorString(action.href));
//     const allowedRoute = allowedByHref.get(href);

//     if (!allowedRoute) {
//       continue;
//     }

//     actions.push({
//       label: getDoctorString(action.label) || allowedRoute.label,
//       href,
//       reason: getDoctorString(action.reason) || allowedRoute.description,
//     });

//     if (actions.length >= 3) {
//       break;
//     }
//   }

//   return actions;
// };

// const formatAIHealthAssistantResponse = (
//   data: Record<string, unknown>,
//   emergencyDetected: boolean,
//   context?: AIHealthApplicationContext,
// ): AIHealthAssistantResponse => {
//   const urgencyLevel = emergencyDetected
//     ? "emergency"
//     : getAIHealthUrgency(data.urgencyLevel);

//   const reply =
//     getDoctorString(data.reply) ||
//     "Please describe the symptoms, duration and severity a little more clearly.";

//   const followUpQuestions = getAIHealthArray(data.followUpQuestions, 3);
//   const suggestedPrompts = getAIHealthArray(data.suggestedPrompts, 4);
//   const allowedRoutes = context?.routes || getAllAIHealthNavigationRoutes();
//   const toolsUsed = context?.toolsUsed.length
//     ? context.toolsUsed
//     : getAIHealthArray(data.toolsUsed, 8);
//   const contextMemoryUsed = context
//     ? context.contextMemoryUsed
//     : getAIHealthBoolean(data.contextMemoryUsed);

//   return {
//     reply,
//     urgencyLevel,
//     suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 3),
//     recommendedActions: getAIHealthArray(data.recommendedActions, 5),
//     warningSigns: emergencyDetected
//       ? Array.from(
//           new Set([
//             "Your description may include an emergency warning sign.",
//             ...getAIHealthArray(data.warningSigns, 4),
//           ]),
//         ).slice(0, 4)
//       : getAIHealthArray(data.warningSigns, 4),
//     followUpQuestions,
//     suggestedPrompts:
//       suggestedPrompts.length > 0
//         ? suggestedPrompts
//         : followUpQuestions.slice(0, 3),
//     navigationActions: getAIHealthNavigationActions(
//       data.navigationActions,
//       allowedRoutes,
//     ),
//     decisionBasis:
//       getDoctorString(data.decisionBasis) ||
//       "This guidance is based on the symptoms, duration, severity, warning signs and relevant SebaSathi application context available in this conversation.",
//     toolsUsed,
//     contextMemoryUsed,
//     disclaimer:
//       getDoctorString(data.disclaimer) ||
//       "General guidance only; this is not a diagnosis or prescription.",
//   };
// };

// const getStoredAIHealthMessages = (value: unknown): AIHealthStoredMessage[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   return value
//     .map((rawMessage): AIHealthStoredMessage | null => {
//       if (typeof rawMessage !== "object" || rawMessage === null) {
//         return null;
//       }

//       const message = rawMessage as Record<string, unknown>;
//       const role = message.role;
//       const content = getDoctorString(message.content);

//       if ((role !== "user" && role !== "assistant") || !content) {
//         return null;
//       }

//       const assistant =
//         typeof message.assistant === "object" && message.assistant !== null
//           ? formatAIHealthAssistantResponse(
//               message.assistant as Record<string, unknown>,
//               false,
//             )
//           : undefined;

//       const createdAtValue = message.createdAt;
//       const createdAt =
//         createdAtValue instanceof Date
//           ? createdAtValue
//           : new Date(
//               typeof createdAtValue === "string" ||
//                 typeof createdAtValue === "number"
//                 ? createdAtValue
//                 : Date.now(),
//             );

//       return {
//         id: getDoctorString(message.id) || createAIHealthMessageId(),
//         role,
//         content,
//         ...(assistant ? { assistant } : {}),
//         createdAt: Number.isNaN(createdAt.getTime()) ? new Date() : createdAt,
//       };
//     })
//     .filter((message): message is AIHealthStoredMessage => message !== null);
// };

// const hasEmergencyWarning = (messages: AIHealthMessage[]): boolean => {
//   const text = messages
//     .filter((message) => message.role === "user")
//     .map((message) => message.content)
//     .join(" ")
//     .toLowerCase();

//   const emergencyPatterns = [
//     /severe chest pain/,
//     /cannot breathe/,
//     /can't breathe/,
//     /difficulty breathing/,
//     /heavy bleeding/,
//     /unconscious/,
//     /not responding/,
//     /seizure/,
//     /stroke symptoms/,
//     /face droop/,
//     /suicid(?:e|al)/,
//     /kill myself/,
//     /বুকে তীব্র ব্যথা/,
//     /শ্বাস নিতে পারছি না/,
//     /শ্বাসকষ্ট/,
//     /অতিরিক্ত রক্তপাত/,
//     /অজ্ঞান/,
//     /খিঁচুনি/,
//     /আত্মহত্যা/,
//   ];

//   return emergencyPatterns.some((pattern) => pattern.test(text));
// };

// const callGroqAI = async (
//   messages: Array<{
//     role: "system" | "user" | "assistant";
//     content: string;
//   }>,
//   temperature: number,
//   maximumOutputTokens: number,
// ): Promise<Record<string, unknown>> => {
//   if (!groqApiKey) {
//     throw new Error("GROQ_API_KEY is missing from the backend .env file");
//   }

//   const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
//     method: "POST",
//     headers: {
//       authorization: `Bearer ${groqApiKey}`,
//       "content-type": "application/json",
//       accept: "application/json",
//     },
//     body: JSON.stringify({
//       model: groqModel,
//       messages,
//       temperature,
//       max_completion_tokens: maximumOutputTokens,
//       response_format: {
//         type: "json_object",
//       },
//     }),
//   });

//   const responseData = (await response.json().catch(() => null)) as Record<
//     string,
//     unknown
//   > | null;

//   if (!response.ok) {
//     const errorObject =
//       typeof responseData?.error === "object" && responseData.error !== null
//         ? (responseData.error as Record<string, unknown>)
//         : null;

//     const providerMessage = getDoctorString(errorObject?.message);

//     throw new Error(
//       providerMessage || `Groq request failed with status ${response.status}`,
//     );
//   }

//   const choices = Array.isArray(responseData?.choices)
//     ? responseData.choices
//     : [];

//   const firstChoice = choices[0];

//   if (typeof firstChoice !== "object" || firstChoice === null) {
//     throw new Error("Groq did not return an assistant response");
//   }

//   const choice = firstChoice as Record<string, unknown>;
//   const message =
//     typeof choice.message === "object" && choice.message !== null
//       ? (choice.message as Record<string, unknown>)
//       : null;

//   const content = getDoctorString(message?.content);

//   if (!content) {
//     throw new Error("Groq returned an empty assistant response");
//   }

//   return extractAIHealthJson(content);
// };

// const callGroqTextStream = async (
//   messages: Array<{
//     role: "system" | "user" | "assistant";
//     content: string;
//   }>,
//   onDelta: (delta: string) => void,
//   signal?: AbortSignal,
// ): Promise<string> => {
//   if (!groqApiKey) {
//     throw new Error("GROQ_API_KEY is missing from the backend .env file");
//   }

//   const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
//     method: "POST",
//     headers: {
//       authorization: `Bearer ${groqApiKey}`,
//       "content-type": "application/json",
//       accept: "text/event-stream",
//     },
//     body: JSON.stringify({
//       model: groqModel,
//       messages,
//       temperature: 0.25,
//       max_completion_tokens: 1100,
//       stream: true,
//     }),
//     signal,
//   });

//   if (!response.ok) {
//     const responseData = (await response.json().catch(() => null)) as Record<
//       string,
//       unknown
//     > | null;
//     const errorObject =
//       typeof responseData?.error === "object" && responseData.error !== null
//         ? (responseData.error as Record<string, unknown>)
//         : null;
//     const providerMessage = getDoctorString(errorObject?.message);

//     throw new Error(
//       providerMessage || `Groq request failed with status ${response.status}`,
//     );
//   }

//   if (!response.body) {
//     throw new Error("Groq streaming response body is unavailable");
//   }

//   const reader = response.body.getReader();
//   const decoder = new TextDecoder();
//   let buffer = "";
//   let completeText = "";

//   while (true) {
//     const { value, done } = await reader.read();

//     if (done) {
//       break;
//     }

//     buffer += decoder.decode(value, { stream: true });
//     const lines = buffer.split("\n");
//     buffer = lines.pop() || "";

//     for (const line of lines) {
//       const trimmed = line.trim();

//       if (!trimmed.startsWith("data:")) {
//         continue;
//       }

//       const payload = trimmed.slice(5).trim();

//       if (!payload || payload === "[DONE]") {
//         continue;
//       }

//       const parsed = JSON.parse(payload) as Record<string, unknown>;
//       const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
//       const firstChoice = choices[0];

//       if (typeof firstChoice !== "object" || firstChoice === null) {
//         continue;
//       }

//       const delta = (firstChoice as Record<string, unknown>).delta;

//       if (typeof delta !== "object" || delta === null) {
//         continue;
//       }

//       const rawContent = (delta as Record<string, unknown>).content;
//       const content = typeof rawContent === "string" ? rawContent : "";

//       if (!content) {
//         continue;
//       }

//       completeText += content;
//       onDelta(content);
//     }
//   }

//   const finalText = completeText.trim();

//   if (!finalText) {
//     throw new Error("Groq returned an empty streamed response");
//   }

//   return finalText;
// };

// const formatAIHealthSummary = (
//   data: Record<string, unknown>,
// ): AIHealthSummaryReport => {
//   return {
//     reportTitle:
//       getDoctorString(data.reportTitle) || "AI Health Conversation Summary",
//     conciseSummary:
//       getDoctorString(data.conciseSummary) ||
//       "A concise summary could not be generated.",
//     chiefConcerns: getAIHealthArray(data.chiefConcerns, 6),
//     symptoms: getAIHealthArray(data.symptoms, 10),
//     durationAndPattern:
//       getDoctorString(data.durationAndPattern) || "Not clearly stated",
//     severity: getDoctorString(data.severity) || "Not clearly stated",
//     urgencyLevel: getAIHealthUrgency(data.urgencyLevel),
//     redFlags: getAIHealthArray(data.redFlags, 6),
//     suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 5),
//     selfCareGuidance: getAIHealthArray(data.selfCareGuidance, 6),
//     questionsForDoctor: getAIHealthArray(data.questionsForDoctor, 6),
//     emergencyAdvice:
//       getDoctorString(data.emergencyAdvice) ||
//       "Seek urgent in-person medical care if symptoms become severe or new warning signs appear.",
//     disclaimer:
//       getDoctorString(data.disclaimer) ||
//       "This AI-generated summary is not a diagnosis or prescription.",
//   };
// };

// const createAIHealthConversationTitle = (message: string): string => {
//   const normalized = message.replace(/\s+/g, " ").trim();
//   const words = normalized.split(" ").filter(Boolean).slice(0, 7);
//   const title = words.join(" ");

//   if (!title) {
//     return "New health chat";
//   }

//   return normalized.length > title.length ? `${title}…` : title;
// };

// const getAIHealthOwnerFilter = (userId: string): Filter<Document> => {
//   return {
//     $or: [
//       {
//         userId,
//       },
//       {
//         patientUserId: userId,
//       },
//     ],
//   };
// };

// const formatAIHealthConversationMessage = (message: AIHealthStoredMessage) => {
//   return {
//     id: message.id,
//     role: message.role,
//     content: message.content,
//     assistant: message.assistant || null,
//     createdAt: formatDoctorDate(message.createdAt),
//   };
// };

// const formatAIHealthConversation = (conversation: Document) => {
//   const userId =
//     getDoctorString(conversation.userId) ||
//     getDoctorString(conversation.patientUserId);

//   const userRole: UserRole =
//     conversation.userRole === "admin" ||
//     conversation.userRole === "doctor" ||
//     conversation.userRole === "patient"
//       ? conversation.userRole
//       : "patient";

//   const messages = getStoredAIHealthMessages(conversation.messages);

//   return {
//     id: getDoctorDocumentId(conversation),
//     title: getDoctorString(conversation.title) || "New health chat",
//     userId,
//     userRole,
//     userName:
//       getDoctorString(conversation.userName) ||
//       getDoctorString(conversation.patientName),
//     userEmail:
//       normalizeDoctorEmail(conversation.userEmail) ||
//       normalizeDoctorEmail(conversation.patientEmail),
//     userImage:
//       getDoctorString(conversation.userImage) ||
//       getDoctorString(conversation.patientImage) ||
//       null,
//     messages: messages.map(formatAIHealthConversationMessage),
//     messageCount: messages.length,
//     summaryHistoryId: getDoctorString(conversation.summaryHistoryId) || null,
//     summaryReport:
//       typeof conversation.summaryReport === "object" &&
//       conversation.summaryReport !== null
//         ? conversation.summaryReport
//         : null,
//     createdAt: formatDoctorDate(conversation.createdAt),
//     updatedAt: formatDoctorDate(conversation.updatedAt),
//     lastMessageAt: formatDoctorDate(
//       conversation.lastMessageAt || conversation.updatedAt,
//     ),
//   };
// };

// const formatAIHealthHistory = (history: Document) => {
//   const userId =
//     getDoctorString(history.userId) || getDoctorString(history.patientUserId);

//   const userName =
//     getDoctorString(history.userName) || getDoctorString(history.patientName);

//   const userEmail =
//     normalizeDoctorEmail(history.userEmail) ||
//     normalizeDoctorEmail(history.patientEmail);

//   const userRole: UserRole =
//     history.userRole === "admin" ||
//     history.userRole === "doctor" ||
//     history.userRole === "patient"
//       ? history.userRole
//       : "patient";

//   return {
//     id: getDoctorDocumentId(history),
//     conversationId: getDoctorString(history.conversationId) || null,
//     conversationTitle: getDoctorString(history.conversationTitle) || null,
//     userId,
//     userRole,
//     userName,
//     userEmail,
//     userImage:
//       getDoctorString(history.userImage) ||
//       getDoctorString(history.patientImage) ||
//       null,
//     patientUserId: userId,
//     patientName: userName,
//     patientEmail: userEmail,
//     provider: getDoctorString(history.provider),
//     model: getDoctorString(history.model),
//     report:
//       typeof history.report === "object" && history.report !== null
//         ? history.report
//         : null,
//     messages: Array.isArray(history.messages) ? history.messages : [],
//     createdAt: formatDoctorDate(history.createdAt),
//     updatedAt: formatDoctorDate(history.updatedAt),
//   };
// };

// const getAIHealthConversationForUser = async (
//   userId: string,
//   conversationId: string,
// ): Promise<Document | null> => {
//   if (!database || !conversationId) {
//     return null;
//   }

//   return database.collection(AI_HEALTH_CHAT_COLLECTION).findOne({
//     $and: [getDoctorFilter(conversationId), getAIHealthOwnerFilter(userId)],
//   });
// };

// const detectAIHealthApplicationIntents = (message: string) => {
//   const normalized = message.toLowerCase();

//   return {
//     appointment:
//       /appointment|booking|schedule|pending|approved|rejected|অ্যাপয়েন্টমেন্ট|অ্যাপয়েন্টমেন্ট|বুকিং|সিডিউল|পেন্ডিং|এপ্রুভ/.test(
//         normalized,
//       ),
//     history:
//       /history|summary|report|previous chat|old chat|হিস্ট্রি|সামারি|রিপোর্ট|পুরোনো চ্যাট/.test(
//         normalized,
//       ),
//     navigation:
//       /open|go to|take me|navigate|where is|show page|dashboard|খুলে দাও|নিয়ে যাও|নিয়ে যাও|কোথায়|কোথায়|ড্যাশবোর্ড/.test(
//         normalized,
//       ),
//     doctor:
//       /doctor|specialist|specialization|cardio|derma|neuro|medicine|surgeon|ডাক্তার|বিশেষজ্ঞ|স্পেশালিস্ট|কার্ডিও|ডার্মা|নিউরো/.test(
//         normalized,
//       ),
//   };
// };

// const buildAIHealthApplicationContext = async (
//   currentUser: Document,
//   latestMessage: string,
//   existingMessages: AIHealthStoredMessage[],
// ): Promise<AIHealthApplicationContext> => {
//   if (!database) {
//     throw new Error("Database is not connected");
//   }

//   const userId = getDoctorDocumentId(currentUser);
//   const role = getNormalizedUserRole(currentUser);
//   const userName = getDoctorString(currentUser.name) || "User";
//   const intents = detectAIHealthApplicationIntents(latestMessage);
//   const routes = getAIHealthNavigationRoutes(role);
//   const toolsUsed = ["SebaSathi navigation map", "SebaSathi doctor directory"];
//   const contextMemoryUsed = existingMessages.length > 0;

//   if (contextMemoryUsed) {
//     toolsUsed.push("Conversation memory");
//   }

//   const [doctorDocuments, specializations, activeDoctorCount] =
//     await Promise.all([
//       database
//         .collection("doctors")
//         .find(
//           { status: "active" },
//           {
//             projection: {
//               name: 1,
//               specialization: 1,
//               hospital: 1,
//               chamber: 1,
//               ratingAverage: 1,
//             },
//           },
//         )
//         .sort({ ratingAverage: -1, ratingCount: -1, createdAt: -1 })
//         .limit(8)
//         .toArray(),
//       database.collection("doctors").distinct("specialization", {
//         status: "active",
//       }),
//       database.collection("doctors").countDocuments({ status: "active" }),
//     ]);

//   const doctorDirectory = {
//     activeDoctorCount,
//     specializations: specializations
//       .map((value) => getDoctorString(value))
//       .filter(Boolean)
//       .sort((a, b) => a.localeCompare(b))
//       .slice(0, 30),
//     highlightedDoctors: doctorDocuments.map((doctor) => ({
//       id: getDoctorDocumentId(doctor),
//       name: getDoctorString(doctor.name),
//       specialization: getDoctorString(doctor.specialization),
//       hospital:
//         getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
//       ratingAverage: Number.isFinite(Number(doctor.ratingAverage))
//         ? Number(Number(doctor.ratingAverage).toFixed(1))
//         : 0,
//     })),
//   };

//   let appointmentContext: AIHealthApplicationContext["appointmentContext"] =
//     null;

//   if (intents.appointment || intents.navigation) {
//     toolsUsed.push("Appointment lookup");

//     const appointmentFilter: Filter<Document> =
//       role === "patient"
//         ? { patientUserId: userId }
//         : role === "doctor"
//           ? { doctorUserId: userId }
//           : {};

//     const [statusCounts, recentAppointments, total] = await Promise.all([
//       database
//         .collection("appointments")
//         .aggregate([
//           { $match: appointmentFilter },
//           {
//             $group: {
//               _id: "$status",
//               count: { $sum: 1 },
//             },
//           },
//         ])
//         .toArray(),
//       database
//         .collection("appointments")
//         .find(appointmentFilter)
//         .sort({ updatedAt: -1, createdAt: -1 })
//         .limit(5)
//         .toArray(),
//       database.collection("appointments").countDocuments(appointmentFilter),
//     ]);

//     appointmentContext = {
//       total,
//       counts: Object.fromEntries(
//         statusCounts.map((item) => [
//           getDoctorString(item._id) || "unknown",
//           Number(item.count) || 0,
//         ]),
//       ),
//       recentAppointments: recentAppointments.map((appointment) => ({
//         id: getDoctorDocumentId(appointment),
//         doctorName: getDoctorString(appointment.doctorName),
//         patientName: getDoctorString(appointment.patientName),
//         specialization: getDoctorString(appointment.specialization),
//         appointmentDate: getDoctorString(appointment.appointmentDate),
//         appointmentTime: getDoctorString(appointment.appointmentTime),
//         status: getDoctorString(appointment.status) || "pending",
//       })),
//     };
//   }

//   let recentHealthHistory: AIHealthApplicationContext["recentHealthHistory"] =
//     [];

//   if (intents.history || intents.navigation) {
//     toolsUsed.push("Saved AI health history lookup");

//     const historyDocuments = await database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .find(getAIHealthOwnerFilter(userId))
//       .sort({ updatedAt: -1, createdAt: -1 })
//       .limit(5)
//       .toArray();

//     recentHealthHistory = historyDocuments.map((history) => {
//       const report =
//         typeof history.report === "object" && history.report !== null
//           ? (history.report as Record<string, unknown>)
//           : {};

//       return {
//         id: getDoctorDocumentId(history),
//         title:
//           getDoctorString(history.conversationTitle) ||
//           getDoctorString(report.reportTitle) ||
//           "AI Health Summary",
//         urgencyLevel: getAIHealthUrgency(report.urgencyLevel),
//         updatedAt: formatDoctorDate(history.updatedAt || history.createdAt),
//       };
//     });
//   }

//   if (intents.doctor) {
//     toolsUsed.push("Specialist matching context");
//   }

//   return {
//     user: {
//       id: userId,
//       name: userName,
//       role,
//     },
//     routes,
//     doctorDirectory,
//     appointmentContext,
//     recentHealthHistory,
//     toolsUsed: Array.from(new Set(toolsUsed)),
//     contextMemoryUsed,
//   };
// };

// const buildAIHealthNaturalResponsePrompt = (
//   context: AIHealthApplicationContext,
// ): string => `You are SebaSathi AI Health Assistant, an advanced conversational assistant integrated into a Bangladesh-oriented healthcare application.

// You must do more than simple text generation. Use conversation memory and the supplied SebaSathi application context to answer questions, reason about next steps, help the user navigate the application, and ask useful follow-up questions when information is missing.

// Signed-in user:
// ${JSON.stringify(context.user)}

// SebaSathi application context retrieved by backend tools:
// ${JSON.stringify({
//   routes: context.routes,
//   doctorDirectory: context.doctorDirectory,
//   appointmentContext: context.appointmentContext,
//   recentHealthHistory: context.recentHealthHistory,
//   toolsUsed: context.toolsUsed,
// })}

// Behavior requirements:
// - Answer health questions and SebaSathi application questions naturally.
// - Use previous conversation messages to understand references such as “it”, “that problem”, “same pain”, or “what should I do next”.
// - When application data is available, use it accurately. Never invent appointments, doctors, history, counts, status, dates or routes.
// - If the user asks where to go in the application, explain the correct page and mention the relevant route label naturally.
// - Explain the practical basis for recommendations without revealing hidden chain-of-thought.
// - Ask concise follow-up questions when key details are missing.
// - Match the user's language: easy Bangla, Banglish or English.
// - For health guidance, never confirm a diagnosis, prescribe medicine, provide individualized doses, or advise stopping prescribed treatment.
// - Emergency warning signs require immediate emergency-care advice.
// - Usually write 5-9 clear sentences and approximately 120-240 words when enough information exists.
// - Return only the natural conversational answer. Do not return JSON, markdown tables, internal IDs or hidden reasoning.`;

// const buildAIHealthMetadataPrompt = (
//   context: AIHealthApplicationContext,
//   latestUserMessage: string,
//   assistantReply: string,
// ): string => `Create safe structured metadata for a completed SebaSathi AI assistant reply.

// User message:
// ${latestUserMessage}

// Assistant reply:
// ${assistantReply}

// Allowed navigation routes:
// ${JSON.stringify(context.routes)}

// Backend tools already used:
// ${JSON.stringify(context.toolsUsed)}

// Return ONLY valid JSON with this exact shape:
// {
//   "urgencyLevel": "routine | soon | urgent | emergency",
//   "suggestedSpecialists": ["maximum three specialist categories that exist in or reasonably map to the doctor directory"],
//   "recommendedActions": ["maximum five safe practical actions"],
//   "warningSigns": ["maximum four important warning signs"],
//   "followUpQuestions": ["maximum three useful follow-up questions"],
//   "suggestedPrompts": ["maximum four short prompts the user can click to continue the conversation"],
//   "navigationActions": [
//     {
//       "label": "must correspond to an allowed route",
//       "href": "must exactly match one allowed route href",
//       "reason": "short explanation of why this page is relevant"
//     }
//   ],
//   "decisionBasis": "one or two concise user-facing sentences explaining which reported facts or application context influenced the guidance, without exposing private chain-of-thought",
//   "toolsUsed": ["copy only tools actually listed above"],
//   "contextMemoryUsed": ${context.contextMemoryUsed ? "true" : "false"},
//   "disclaimer": "one short medical disclaimer"
// }

// Do not invent app data. Include navigationActions only when useful. Suggested prompts must be directly usable as the user's next message.`;

// const writeAIHealthStreamEvent = (
//   res: Response,
//   event: Record<string, unknown>,
// ): void => {
//   if (!res.writableEnded) {
//     res.write(`${JSON.stringify(event)}\n`);
//   }
// };

// const startAIHealthStream = (res: Response): void => {
//   res.status(200);
//   res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
//   res.setHeader("Cache-Control", "no-cache, no-transform");
//   res.setHeader("Connection", "keep-alive");
//   res.setHeader("X-Accel-Buffering", "no");
//   res.flushHeaders();
// };

// const writeAIHealthStatus = (
//   res: Response,
//   stage: AIHealthStreamStage,
//   message: string,
//   toolsUsed: string[] = [],
// ): void => {
//   writeAIHealthStreamEvent(res, {
//     type: "status",
//     stage,
//     message,
//     toolsUsed,
//   });
// };

// /* =========================================================
//    AI Health access status
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/access",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           allowed: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);
//       const allowed = status === "active";

//       res.status(200).json({
//         success: true,
//         authenticated: true,
//         allowed,
//         role,
//         status,
//         user: {
//           id: getDoctorDocumentId(currentUser),
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//         },
//         message: allowed
//           ? "Your active account can use SebaSathi AI Health Assistant."
//           : "Your account is blocked. Contact the administrator to use the AI Health Assistant.",
//       });
//     } catch (error) {
//       console.error("AI Health access error:", error);

//       res.status(500).json({
//         success: false,
//         allowed: false,
//         message: "Failed to verify AI Health access",
//       });
//     }
//   },
// );

// /* =========================================================
//    AI Health conversation history
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/conversations",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const limit = getPositiveInteger(req.query.limit, 100, 100);
//       const conversations = await database
//         .collection(AI_HEALTH_CHAT_COLLECTION)
//         .find(getAIHealthOwnerFilter(userId))
//         .sort({
//           lastMessageAt: -1,
//           updatedAt: -1,
//           _id: -1,
//         })
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         conversations: conversations.map(formatAIHealthConversation),
//       });
//     } catch (error) {
//       console.error("Get AI conversations error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI conversations",
//       });
//     }
//   },
// );

// app.post(
//   "/api/v1/ai-health/conversations",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const now = new Date();
//       const userId = getDoctorDocumentId(currentUser);
//       const userRole = getNormalizedUserRole(currentUser);
//       const userName = getDoctorString(currentUser.name);
//       const userEmail = normalizeDoctorEmail(currentUser.email);
//       const userImage = getDoctorString(currentUser.image) || null;
//       const requestedTitle = getDoctorString(req.body.title).slice(0, 80);

//       const conversationDocument = {
//         title: requestedTitle || "New health chat",
//         userId,
//         userRole,
//         userName,
//         userEmail,
//         userImage,
//         patientUserId: userId,
//         patientName: userName,
//         patientEmail: userEmail,
//         patientImage: userImage,
//         messages: [] as AIHealthStoredMessage[],
//         summaryHistoryId: null,
//         summaryReport: null,
//         createdAt: now,
//         updatedAt: now,
//         lastMessageAt: now,
//       };

//       const collection = database.collection(AI_HEALTH_CHAT_COLLECTION);
//       const insertResult = await collection.insertOne(conversationDocument);
//       const conversation = await collection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "New AI health conversation created",
//         conversation: conversation
//           ? formatAIHealthConversation(conversation)
//           : {
//               id: insertResult.insertedId.toHexString(),
//               ...conversationDocument,
//               messageCount: 0,
//             },
//       });
//     } catch (error) {
//       console.error("Create AI conversation error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to create AI conversation",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/ai-health/conversations/:conversationId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         getDoctorDocumentId(currentUser),
//         getDoctorString(req.params.conversationId),
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         conversation: formatAIHealthConversation(conversation),
//       });
//     } catch (error) {
//       console.error("Get AI conversation details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI conversation",
//       });
//     }
//   },
// );

// app.delete(
//   "/api/v1/ai-health/conversations/:conversationId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       await database.collection(AI_HEALTH_CHAT_COLLECTION).deleteOne({
//         _id: conversation._id,
//       });

//       res.status(200).json({
//         success: true,
//         message: "AI health conversation deleted successfully",
//         deletedConversationId: conversationId,
//       });
//     } catch (error) {
//       console.error("Delete AI conversation error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to delete AI conversation",
//       });
//     }
//   },
// );

// /* =========================================================
//    Advanced streamed AI Health message exchange
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/conversations/:conversationId/messages/stream",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     let streamStarted = false;
//     const abortController = new AbortController();

//     res.on("close", () => {
//       if (!res.writableEnded) {
//         abortController.abort();
//       }
//     });

//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const content = getDoctorString(req.body.message);

//       if (!content) {
//         res.status(400).json({
//           success: false,
//           message: "A health or application question is required",
//         });
//         return;
//       }

//       if (content.length > 4000) {
//         res.status(400).json({
//           success: false,
//           message: "A message cannot contain more than 4000 characters",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       const existingMessages = getStoredAIHealthMessages(conversation.messages);
//       const contextMessages: AIHealthMessage[] = [
//         ...existingMessages.map(({ role, content: savedContent }) => ({
//           role,
//           content: savedContent,
//         })),
//         {
//           role: "user",
//           content,
//         },
//       ];

//       const validation = normalizeAIHealthMessages(contextMessages, {
//         requireLatestUser: true,
//         maximumMessages: 26,
//         maximumCharacters: 32000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       startAIHealthStream(res);
//       streamStarted = true;
//       writeAIHealthStatus(
//         res,
//         "thinking",
//         "Understanding your question and previous conversation...",
//       );

//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         content,
//         existingMessages,
//       );

//       writeAIHealthStatus(
//         res,
//         "tool",
//         "Checking relevant SebaSathi context...",
//         applicationContext.toolsUsed,
//       );

//       writeAIHealthStatus(
//         res,
//         "answering",
//         "Preparing a context-aware response...",
//         applicationContext.toolsUsed,
//       );

//       const naturalReply = await callGroqTextStream(
//         [
//           {
//             role: "system",
//             content: buildAIHealthNaturalResponsePrompt(applicationContext),
//           },
//           ...validation.messages,
//         ],
//         (delta) => {
//           writeAIHealthStreamEvent(res, {
//             type: "delta",
//             delta,
//           });
//         },
//         abortController.signal,
//       );

//       writeAIHealthStatus(
//         res,
//         "structuring",
//         "Creating follow-up prompts, navigation actions and decision support...",
//         applicationContext.toolsUsed,
//       );

//       let metadata: Record<string, unknown> = {};

//       try {
//         metadata = await callGroqAI(
//           [
//             {
//               role: "system",
//               content: buildAIHealthMetadataPrompt(
//                 applicationContext,
//                 content,
//                 naturalReply,
//               ),
//             },
//             {
//               role: "user",
//               content: "Return the requested JSON metadata now.",
//             },
//           ],
//           0.1,
//           900,
//         );
//       } catch (metadataError) {
//         console.error(
//           "AI Health metadata generation warning:",
//           metadataError instanceof Error
//             ? metadataError.message
//             : metadataError,
//         );
//       }

//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const assistant = formatAIHealthAssistantResponse(
//         {
//           ...metadata,
//           reply: naturalReply,
//           toolsUsed: applicationContext.toolsUsed,
//           contextMemoryUsed: applicationContext.contextMemoryUsed,
//         },
//         emergencyDetected,
//         applicationContext,
//       );

//       const now = new Date();
//       const userMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "user",
//         content,
//         createdAt: now,
//       };
//       const assistantMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "assistant",
//         content: naturalReply,
//         assistant,
//         createdAt: new Date(),
//       };

//       const nextTitle =
//         existingMessages.some((message) => message.role === "user") ||
//         getDoctorString(conversation.title) !== "New health chat"
//           ? getDoctorString(conversation.title) || "New health chat"
//           : createAIHealthConversationTitle(content);

//       writeAIHealthStatus(
//         res,
//         "saving",
//         "Saving the conversation and memory...",
//         applicationContext.toolsUsed,
//       );

//       const updatedConversation = await database
//         .collection<AIHealthConversationDocument>(AI_HEALTH_CHAT_COLLECTION)
//         .findOneAndUpdate(
//           {
//             _id: conversation._id,
//           },
//           {
//             $set: {
//               title: nextTitle,
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: assistantMessage.createdAt,
//               lastMessageAt: assistantMessage.createdAt,
//             },
//             $push: {
//               messages: {
//                 $each: [userMessage, assistantMessage],
//               },
//             },
//           },
//           {
//             returnDocument: "after",
//           },
//         );

//       writeAIHealthStreamEvent(res, {
//         type: "result",
//         data: {
//           success: true,
//           provider: "groq",
//           model: groqModel,
//           userMessage: formatAIHealthConversationMessage(userMessage),
//           assistantMessage: formatAIHealthConversationMessage(assistantMessage),
//           conversation: updatedConversation
//             ? formatAIHealthConversation(updatedConversation)
//             : null,
//         },
//       });

//       res.end();
//     } catch (error) {
//       console.error("AI Health streamed chat error:", error);

//       const message =
//         error instanceof Error
//           ? error.message
//           : "Failed to receive a streamed response from the AI provider";

//       if (streamStarted) {
//         writeAIHealthStreamEvent(res, {
//           type: "error",
//           message,
//         });
//         res.end();
//       } else {
//         res.status(502).json({
//           success: false,
//           message,
//         });
//       }
//     }
//   },
// );

// /* =========================================================
//    Non-streaming persistent message exchange compatibility
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/conversations/:conversationId/messages",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const content = getDoctorString(req.body.message);

//       if (!content) {
//         res.status(400).json({
//           success: false,
//           message: "A health or application question is required",
//         });
//         return;
//       }

//       if (content.length > 4000) {
//         res.status(400).json({
//           success: false,
//           message: "A message cannot contain more than 4000 characters",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       const existingMessages = getStoredAIHealthMessages(conversation.messages);
//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         content,
//         existingMessages,
//       );
//       const contextMessages: AIHealthMessage[] = [
//         ...existingMessages.map(({ role, content: savedContent }) => ({
//           role,
//           content: savedContent,
//         })),
//         {
//           role: "user",
//           content,
//         },
//       ];
//       const validation = normalizeAIHealthMessages(contextMessages, {
//         requireLatestUser: true,
//         maximumMessages: 26,
//         maximumCharacters: 32000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const groqData = await callGroqAI(
//         [
//           {
//             role: "system",
//             content: `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn ONLY JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`,
//           },
//           ...validation.messages,
//         ],
//         0.2,
//         1200,
//       );

//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const assistant = formatAIHealthAssistantResponse(
//         groqData,
//         emergencyDetected,
//         applicationContext,
//       );
//       const now = new Date();
//       const userMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "user",
//         content,
//         createdAt: now,
//       };
//       const assistantMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "assistant",
//         content: assistant.reply,
//         assistant,
//         createdAt: new Date(),
//       };
//       const nextTitle =
//         existingMessages.some((message) => message.role === "user") ||
//         getDoctorString(conversation.title) !== "New health chat"
//           ? getDoctorString(conversation.title) || "New health chat"
//           : createAIHealthConversationTitle(content);

//       const updatedConversation = await database
//         .collection<AIHealthConversationDocument>(AI_HEALTH_CHAT_COLLECTION)
//         .findOneAndUpdate(
//           { _id: conversation._id },
//           {
//             $set: {
//               title: nextTitle,
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: assistantMessage.createdAt,
//               lastMessageAt: assistantMessage.createdAt,
//             },
//             $push: {
//               messages: {
//                 $each: [userMessage, assistantMessage],
//               },
//             },
//           },
//           { returnDocument: "after" },
//         );

//       res.status(200).json({
//         success: true,
//         provider: "groq",
//         model: groqModel,
//         userMessage: formatAIHealthConversationMessage(userMessage),
//         assistantMessage: formatAIHealthConversationMessage(assistantMessage),
//         conversation: updatedConversation
//           ? formatAIHealthConversation(updatedConversation)
//           : null,
//       });
//     } catch (error) {
//       console.error("AI Health persistent chat error:", error);

//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to receive a response from the AI provider",
//       });
//     }
//   },
// );

// /* =========================================================
//    Legacy AI Health chat endpoint
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/chat",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const validation = normalizeAIHealthMessages(req.body.messages, {
//         requireLatestUser: true,
//         maximumMessages: 22,
//         maximumCharacters: 26000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const latestMessage = validation.messages.at(-1)?.content || "";
//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         latestMessage,
//         [],
//       );
//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const systemPrompt = `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn only JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`;

//       const groqData = await callGroqAI(
//         [{ role: "system", content: systemPrompt }, ...validation.messages],
//         0.2,
//         1200,
//       );

//       res.status(200).json({
//         success: true,
//         provider: "groq",
//         model: groqModel,
//         assistant: formatAIHealthAssistantResponse(
//           groqData,
//           emergencyDetected,
//           applicationContext,
//         ),
//       });
//     } catch (error) {
//       console.error("AI Health chat error:", error);
//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to receive a response from the AI provider",
//       });
//     }
//   },
// );

// /* =========================================================
//    Generate and save AI Health summary
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/summary",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.body.conversationId);
//       let conversation: Document | null = null;
//       let conversationTitle = "AI Health Conversation";
//       let messagesValue: unknown = req.body.messages;

//       if (conversationId) {
//         conversation = await getAIHealthConversationForUser(
//           userId,
//           conversationId,
//         );

//         if (!conversation) {
//           res.status(404).json({
//             success: false,
//             message: "AI health conversation was not found",
//           });
//           return;
//         }

//         conversationTitle =
//           getDoctorString(conversation.title) || "AI Health Conversation";
//         messagesValue = getStoredAIHealthMessages(conversation.messages).map(
//           ({ role, content }) => ({ role, content }),
//         );
//       }

//       const validation = normalizeAIHealthMessages(messagesValue, {
//         requireLatestUser: false,
//         maximumMessages: 40,
//         maximumCharacters: 42000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const systemPrompt = `Generate a concise structured health-conversation report for SebaSathi AI. Use only information actually present. Do not invent symptoms, duration, tests, diagnoses or medicines. Do not diagnose or prescribe. Match the user's language where practical.

// Return ONLY valid JSON:
// {
//   "reportTitle": "short title",
//   "conciseSummary": "2-3 concise sentences",
//   "chiefConcerns": ["main concerns"],
//   "symptoms": ["reported symptoms"],
//   "durationAndPattern": "stated duration/pattern or Not clearly stated",
//   "severity": "stated severity or Not clearly stated",
//   "urgencyLevel": "routine | soon | urgent | emergency",
//   "redFlags": ["warning signs"],
//   "suggestedSpecialists": ["specialist categories"],
//   "selfCareGuidance": ["low-risk general guidance"],
//   "questionsForDoctor": ["useful questions"],
//   "emergencyAdvice": "brief emergency advice",
//   "disclaimer": "not a diagnosis or prescription"
// }`;

//       const groqData = await callGroqAI(
//         [
//           {
//             role: "system",
//             content: systemPrompt,
//           },
//           {
//             role: "user",
//             content: JSON.stringify(validation.messages),
//           },
//         ],
//         0.1,
//         1100,
//       );

//       const report = formatAIHealthSummary(groqData);
//       const now = new Date();
//       const userRole = getNormalizedUserRole(currentUser);
//       const userName = getDoctorString(currentUser.name);
//       const userEmail = normalizeDoctorEmail(currentUser.email);
//       const userImage = getDoctorString(currentUser.image) || null;
//       const historyCollection = database.collection(
//         AI_HEALTH_HISTORY_COLLECTION,
//       );

//       const historyDocument = {
//         conversationId: conversation ? getDoctorDocumentId(conversation) : null,
//         conversationTitle,
//         userId,
//         userRole,
//         userName,
//         userEmail,
//         userImage,
//         patientUserId: userId,
//         patientName: userName,
//         patientEmail: userEmail,
//         patientImage: userImage,
//         provider: "groq",
//         model: groqModel,
//         report,
//         messages: validation.messages,
//         createdAt: now,
//         updatedAt: now,
//       };

//       let history: Document | null = null;
//       const existingSummaryId = getDoctorString(conversation?.summaryHistoryId);

//       if (existingSummaryId) {
//         const existingHistory = await historyCollection.findOne({
//           $and: [
//             getDoctorFilter(existingSummaryId),
//             getAIHealthOwnerFilter(userId),
//           ],
//         });

//         if (existingHistory) {
//           history = await historyCollection.findOneAndUpdate(
//             { _id: existingHistory._id },
//             {
//               $set: {
//                 ...historyDocument,
//                 createdAt: existingHistory.createdAt || now,
//                 updatedAt: now,
//               },
//             },
//             { returnDocument: "after" },
//           );
//         }
//       }

//       if (!history) {
//         const insertResult = await historyCollection.insertOne(historyDocument);
//         history = await historyCollection.findOne({
//           _id: insertResult.insertedId,
//         });
//       }

//       if (!history) {
//         throw new Error("The generated summary could not be saved");
//       }

//       if (conversation) {
//         await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
//           { _id: conversation._id },
//           {
//             $set: {
//               summaryHistoryId: getDoctorDocumentId(history),
//               summaryReport: report,
//               updatedAt: now,
//             },
//           },
//         );
//       }

//       res.status(201).json({
//         success: true,
//         message: "AI health summary generated and saved successfully",
//         history: formatAIHealthHistory(history),
//         conversation: conversation
//           ? formatAIHealthConversation({
//               ...conversation,
//               summaryHistoryId: getDoctorDocumentId(history),
//               summaryReport: report,
//               updatedAt: now,
//             })
//           : null,
//       });
//     } catch (error) {
//       console.error("AI Health summary error:", error);

//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to generate and save the AI health summary",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient AI Health summary history
//    - Active and blocked patients can read their own history.
//    - Only active patients can delete their own history.
// ========================================================= */

// app.get(
//   "/api/v1/patient/ai-health-history",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const status = getNormalizedUserStatus(currentUser);
//       const requestedPage = getPositiveInteger(req.query.page, 1, 100000);

//       // Patient AI Health History always returns exactly 10 records per page.
//       const limit = 10;
//       const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
//       const filter = getAIHealthOwnerFilter(patientUserId);
//       const total = await collection.countDocuments(filter);
//       const totalPages = Math.max(1, Math.ceil(total / limit));
//       const page = Math.min(requestedPage, totalPages);

//       const documents = await collection
//         .find(filter)
//         .sort({
//           updatedAt: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .skip((page - 1) * limit)
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         account: {
//           id: patientUserId,
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//           role: "patient",
//           status,
//         },
//         canDelete: status === "active",
//         histories: documents.map(formatAIHealthHistory),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages,
//         },
//       });
//     } catch (error) {
//       console.error("Get patient AI Health history error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient AI health history",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/patient/ai-health-history/:historyId",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);

//       if (!historyId) {
//         res.status(400).json({
//           success: false,
//           message: "AI health history ID is required",
//         });
//         return;
//       }

//       const history = await database
//         .collection(AI_HEALTH_HISTORY_COLLECTION)
//         .findOne({
//           $and: [
//             getDoctorFilter(historyId),
//             getAIHealthOwnerFilter(patientUserId),
//           ],
//         });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       const status = getNormalizedUserStatus(currentUser);

//       res.status(200).json({
//         success: true,
//         account: {
//           id: patientUserId,
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//           role: "patient",
//           status,
//         },
//         canDelete: status === "active",
//         history: formatAIHealthHistory(history),
//       });
//     } catch (error) {
//       console.error("Get patient AI Health history details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient AI health history details",
//       });
//     }
//   },
// );

// app.delete(
//   "/api/v1/patient/ai-health-history/:historyId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);

//       if (!historyId) {
//         res.status(400).json({
//           success: false,
//           message: "AI health history ID is required",
//         });
//         return;
//       }

//       const historyCollection = database.collection(
//         AI_HEALTH_HISTORY_COLLECTION,
//       );

//       const history = await historyCollection.findOne({
//         $and: [
//           getDoctorFilter(historyId),
//           getAIHealthOwnerFilter(patientUserId),
//         ],
//       });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       const deleteResult = await historyCollection.deleteOne({
//         _id: history._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "AI health history could not be deleted",
//         });
//         return;
//       }

//       const conversationId = getDoctorString(history.conversationId);

//       if (conversationId) {
//         await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
//           {
//             $and: [
//               getDoctorFilter(conversationId),
//               getAIHealthOwnerFilter(patientUserId),
//               {
//                 summaryHistoryId: historyId,
//               },
//             ],
//           },
//           {
//             $set: {
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: new Date(),
//             },
//           },
//         );
//       }

//       res.status(200).json({
//         success: true,
//         message: "AI health history deleted successfully",
//         deletedHistoryId: historyId,
//       });
//     } catch (error) {
//       console.error("Delete patient AI Health history error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete patient AI health history",
//       });
//     }
//   },
// );

// /* =========================================================
//    Saved AI Health summary history
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/history",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;
//       const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
//       const filter = getAIHealthOwnerFilter(userId);

//       const [documents, total] = await Promise.all([
//         collection
//           .find(filter)
//           .sort({ createdAt: -1, _id: -1 })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),
//         collection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         histories: documents.map(formatAIHealthHistory),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get AI Health history error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI health history",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/ai-health/history/:historyId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);
//       const history = await database
//         .collection(AI_HEALTH_HISTORY_COLLECTION)
//         .findOne({
//           $and: [getDoctorFilter(historyId), getAIHealthOwnerFilter(userId)],
//         });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         history: formatAIHealthHistory(history),
//       });
//     } catch (error) {
//       console.error("Get AI Health history details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI health history details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Unknown route handler
// ========================================================= */

// app.use((_req: Request, res: Response) => {
//   res.status(404).json({
//     success: false,
//     message: "API route not found",
//   });
// });

// /* =========================================================
//    Global error handler
// ========================================================= */

// app.use((error: Error, _req: Request, res: Response, _next: NextFunction) => {
//   console.error("Server error:", error);

//   res.status(500).json({
//     success: false,
//     message: "Internal server error",
//   });
// });

// /* =========================================================
//    MongoDB connection
// ========================================================= */

// const connectDatabase = async (): Promise<void> => {
//   await mongoClient.connect();

//   database = mongoClient.db(mongoDbName);

//   await database.command({ ping: 1 });

//   await Promise.all([
//     database
//       .collection("user")
//       .createIndex({ role: 1, status: 1, updatedAt: -1 }),
//     database.collection("user").createIndex({ role: 1, name: 1 }),
//     database.collection("user").createIndex({ role: 1, email: 1 }),
//     database
//       .collection("doctors")
//       .createIndex({ status: 1, ratingAverage: -1, createdAt: -1 }),
//     database.collection("doctors").createIndex({ name: 1 }),
//     database.collection("doctors").createIndex({ specialization: 1 }),
//     database.collection("doctors").createIndex({ qualification: 1 }),
//     database.collection("doctors").createIndex({ hospital: 1 }),
//     database.collection("doctors").createIndex({ experienceYears: 1 }),
//     database
//       .collection("reviews")
//       .createIndex({ doctorId: 1, userId: 1 }, { unique: true }),
//     database.collection("reviews").createIndex({ doctorId: 1, updatedAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ patientUserId: 1, doctorId: 1, status: 1 }),
//     database
//       .collection("appointments")
//       .createIndex({ patientUserId: 1, createdAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ doctorUserId: 1, status: 1, appointmentDate: 1 }),
//     database
//       .collection("appointments")
//       .createIndex({ doctorUserId: 1, createdAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ status: 1, appointmentDate: 1, appointmentTime: 1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ userId: 1, createdAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ patientUserId: 1, createdAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ userId: 1, updatedAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ patientUserId: 1, updatedAt: -1 }),
//     database
//       .collection(AI_HEALTH_CHAT_COLLECTION)
//       .createIndex({ userId: 1, lastMessageAt: -1 }),
//     database
//       .collection(AI_HEALTH_CHAT_COLLECTION)
//       .createIndex({ patientUserId: 1, lastMessageAt: -1 }),
//   ]);

//   console.log(`MongoDB connected successfully. Database: ${mongoDbName}`);
// };

// /* =========================================================
//    Start server
// ========================================================= */

// const startServer = async (): Promise<void> => {
//   try {
//     await connectDatabase();

//     app.listen(port, () => {
//       console.log(`SebaSathi AI server is running on http://localhost:${port}`);

//       console.log(`JWKS URL: ${jwksUrl.toString()}`);
//     });
//   } catch (error) {
//     console.error(
//       "Unable to start SebaSathi AI server:",
//       error instanceof Error ? error.message : error,
//     );

//     await mongoClient.close();
//     process.exit(1);
//   }
// };

// void startServer();

// /* =========================================================
//    Graceful shutdown
// ========================================================= */

// const shutdownServer = async (signal: string): Promise<void> => {
//   console.log(`${signal} received. Closing MongoDB connection...`);

//   try {
//     await mongoClient.close();

//     console.log("MongoDB connection closed successfully");

//     process.exit(0);
//   } catch (error) {
//     console.error("Error closing MongoDB connection:", error);

//     process.exit(1);
//   }
// };

// process.on("SIGINT", () => {
//   void shutdownServer("SIGINT");
// });

// process.on("SIGTERM", () => {
//   void shutdownServer("SIGTERM");
// });

// export default app;

//===================================== 21july

// import cors from "cors";
// import dotenv from "dotenv";
// import express, {
//   type NextFunction,
//   type Request,
//   type Response,
// } from "express";
// import { createRemoteJWKSet, jwtVerify } from "jose-cjs";
// import {
//   MongoClient,
//   ObjectId,
//   ServerApiVersion,
//   type Db,
//   type Document,
//   type Filter,
// } from "mongodb";

// dotenv.config({ quiet: true });

// /* =========================================================
//    Environment variables
// ========================================================= */

// const port = Number(process.env.PORT) || 5000;

// const clientUrl = process.env.CLIENT_URL || "http://localhost:3000";

// const betterAuthUrl = (
//   process.env.BETTER_AUTH_URL || "http://localhost:3000"
// ).replace(/\/+$/, "");

// const mongoDbUri = process.env.MONGODB_URI;
// const mongoDbName = process.env.MONGODB_DB_NAME;

// const groqApiKey = process.env.GROQ_API_KEY;

// const groqApiBaseUrl = (
//   process.env.GROQ_API_BASE_URL || "https://api.groq.com/openai/v1"
// ).replace(/\/+$/, "");

// const groqModel = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

// const AI_HEALTH_HISTORY_COLLECTION = "AI-health-History";

// const AI_HEALTH_CHAT_COLLECTION = "all-history";

// if (!mongoDbUri) {
//   throw new Error("MONGODB_URI is missing from the .env file");
// }

// if (!mongoDbName) {
//   throw new Error("MONGODB_DB_NAME is missing from the .env file");
// }

// /* =========================================================
//    Express application
// ========================================================= */

// const app = express();

// /* =========================================================
//    MongoDB configuration
// ========================================================= */

// const mongoClient = new MongoClient(mongoDbUri, {
//   serverApi: {
//     version: ServerApiVersion.v1,
//     strict: false,
//     deprecationErrors: true,
//   },
// });

// let database: Db | null = null;

// /* =========================================================
//    Better Auth JWKS configuration
// ========================================================= */

// const jwksUrl = new URL(`${betterAuthUrl}/api/auth/jwks`);

// const jwks = createRemoteJWKSet(jwksUrl);

// /* =========================================================
//    Authentication types
// ========================================================= */

// type UserRole = "admin" | "doctor" | "patient";
// type UserStatus = "active" | "blocked";

// interface AuthenticatedRequest extends Request {
//   userId?: string;
//   userName?: string;
//   userEmail?: string;
//   userRole?: UserRole;
//   userStatus?: UserStatus;
// }

// /* =========================================================
//    Global middlewares
// ========================================================= */

// app.use(
//   cors({
//     origin: clientUrl,
//     credentials: true,
//   }),
// );

// app.use(
//   express.json({
//     limit: "1mb",
//   }),
// );

// app.use(express.urlencoded({ extended: true }));

// /* =========================================================
//    JWT verification middleware
// ========================================================= */

// const verifyToken = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): Promise<void> => {
//   const authorizationHeader = req.headers.authorization;

//   if (!authorizationHeader) {
//     res.status(401).json({
//       success: false,
//       message: "Authorization token is required",
//     });

//     return;
//   }

//   const [authorizationType, token] = authorizationHeader.split(" ");

//   if (authorizationType !== "Bearer" || !token) {
//     res.status(401).json({
//       success: false,
//       message: "A valid Bearer token is required",
//     });

//     return;
//   }

//   try {
//     const { payload } = await jwtVerify(token, jwks);

//     const authenticatedUserId =
//       typeof payload.sub === "string"
//         ? payload.sub
//         : typeof payload.id === "string"
//           ? payload.id
//           : undefined;

//     if (!authenticatedUserId) {
//       res.status(403).json({
//         success: false,
//         message: "Token does not contain a valid user ID",
//       });

//       return;
//     }

//     req.userId = authenticatedUserId;

//     req.userName = typeof payload.name === "string" ? payload.name : undefined;

//     req.userEmail =
//       typeof payload.email === "string" ? payload.email : undefined;

//     next();
//   } catch (error) {
//     console.error(
//       "JWT verification error:",
//       error instanceof Error ? error.message : error,
//     );

//     res.status(403).json({
//       success: false,
//       message: "Invalid or expired access token",
//     });
//   }
// };

// /* =========================================================
//    Role verification middleware
// ========================================================= */

// const verifyRole = (requiredRole: UserRole) => {
//   return async (
//     req: AuthenticatedRequest,
//     res: Response,
//     next: NextFunction,
//   ): Promise<void> => {
//     try {
//       if (!req.userId) {
//         res.status(401).json({
//           success: false,
//           message: "Authentication is required before role verification",
//         });

//         return;
//       }

//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const userQueryConditions: Record<string, unknown>[] = [
//         {
//           id: req.userId,
//         },
//       ];

//       if (req.userEmail) {
//         userQueryConditions.push({
//           email: req.userEmail,
//         });
//       }

//       const currentUser = await usersCollection.findOne({
//         $or: userQueryConditions,
//       });

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       req.userStatus = currentUser.status === "blocked" ? "blocked" : "active";

//       const currentRole = currentUser.role;

//       const validRoles: UserRole[] = ["admin", "doctor", "patient"];

//       if (
//         typeof currentRole !== "string" ||
//         !validRoles.includes(currentRole as UserRole)
//       ) {
//         res.status(403).json({
//           success: false,
//           message: "User role is missing or invalid",
//         });

//         return;
//       }

//       if (currentRole !== requiredRole) {
//         res.status(403).json({
//           success: false,
//           message: `${requiredRole} access is required`,
//         });

//         return;
//       }

//       req.userRole = currentRole as UserRole;

//       next();
//     } catch (error) {
//       console.error(
//         "Role verification error:",
//         error instanceof Error ? error.message : error,
//       );

//       res.status(500).json({
//         success: false,
//         message: "Failed to verify current user role",
//       });
//     }
//   };
// };

// /* =========================================================
//    Admin, doctor and patient middlewares
// ========================================================= */

// const verifyAdmin = verifyRole("admin");

// const verifyDoctor = verifyRole("doctor");

// const verifyPatient = verifyRole("patient");

// /**
//  * Allows blocked users to read data, but prevents them
//  * from creating, editing, deleting, or changing status.
//  *
//  * Always use this after verifyRole().
//  */
// const verifyActive = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): void => {
//   if (req.userStatus !== "active") {
//     res.status(403).json({
//       success: false,
//       message:
//         "Your account is blocked. You can view data, but you cannot perform this action.",
//       code: "READ_ONLY_ACCOUNT",
//     });

//     return;
//   }

//   next();
// };

// /**
//  * Allows any authenticated role (admin, doctor or patient)
//  * to use protected features when the account status is active.
//  */
// const verifyAnyActiveUser = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): Promise<void> => {
//   try {
//     if (!req.userId) {
//       res.status(401).json({
//         success: false,
//         message: "Authentication is required",
//       });

//       return;
//     }

//     if (!database) {
//       res.status(503).json({
//         success: false,
//         message: "Database is not connected",
//       });

//       return;
//     }

//     const userQueryConditions: Record<string, unknown>[] = [
//       {
//         id: req.userId,
//       },
//     ];

//     if (req.userEmail) {
//       userQueryConditions.push({
//         email: req.userEmail.toLowerCase(),
//       });
//     }

//     const currentUser = await database.collection("user").findOne({
//       $or: userQueryConditions,
//     });

//     if (!currentUser) {
//       res.status(404).json({
//         success: false,
//         message: "User account was not found",
//       });

//       return;
//     }

//     const currentRole = currentUser.role;
//     const validRoles: UserRole[] = ["admin", "doctor", "patient"];

//     if (
//       typeof currentRole !== "string" ||
//       !validRoles.includes(currentRole as UserRole)
//     ) {
//       res.status(403).json({
//         success: false,
//         message: "User role is missing or invalid",
//       });

//       return;
//     }

//     const currentStatus: UserStatus =
//       currentUser.status === "blocked" ? "blocked" : "active";

//     req.userRole = currentRole as UserRole;
//     req.userStatus = currentStatus;

//     if (currentStatus !== "active") {
//       res.status(403).json({
//         success: false,
//         message:
//           "Your account is blocked. Only active accounts can use the AI Health Assistant.",
//         code: "READ_ONLY_ACCOUNT",
//       });

//       return;
//     }

//     next();
//   } catch (error) {
//     console.error(
//       "Active user verification error:",
//       error instanceof Error ? error.message : error,
//     );

//     res.status(500).json({
//       success: false,
//       message: "Failed to verify active user account",
//     });
//   }
// };

// /* =========================================================
//    Public root route
// ========================================================= */

// app.get("/", (_req: Request, res: Response) => {
//   res.status(200).json({
//     success: true,
//     message: "SebaSathi AI server is running",
//   });
// });

// /* =========================================================
//    Public health route
// ========================================================= */

// app.get("/api/v1/health", async (_req: Request, res: Response) => {
//   try {
//     if (!database) {
//       res.status(503).json({
//         success: false,
//         message: "Database is not connected",
//       });

//       return;
//     }

//     await database.command({ ping: 1 });

//     res.status(200).json({
//       success: true,
//       message: "SebaSathi AI API is healthy",
//       database: "connected",
//       databaseName: database.databaseName,
//       timestamp: new Date().toISOString(),
//     });
//   } catch {
//     res.status(503).json({
//       success: false,
//       message: "MongoDB connection is unavailable",
//       database: "disconnected",
//       timestamp: new Date().toISOString(),
//     });
//   }
// });

// /* =========================================================
//    Protected authentication test route
// ========================================================= */

// app.get(
//   "/api/v1/auth/me",
//   verifyToken,
//   (req: AuthenticatedRequest, res: Response) => {
//     res.status(200).json({
//       success: true,
//       message: "Authenticated user retrieved successfully",
//       user: {
//         id: req.userId,
//         name: req.userName || null,
//         email: req.userEmail || null,
//       },
//     });
//   },
// );

// /*
//   Admin API middleware:

//   app.get(
//     "/api/v1/admin/your-api",
//     verifyToken,
//     verifyAdmin,
//     yourController
//   );
// */

// /*
//   Doctor API middleware:

//   app.get(
//     "/api/v1/doctor/your-api",
//     verifyToken,
//     verifyDoctor,
//     yourController
//   );
// */

// /*
//   Patient API middleware:

//   app.get(
//     "/api/v1/patient/your-api",
//     verifyToken,
//     verifyPatient,
//     yourController
//   );
// */

// /* =========================================================
//    Current authenticated user
// ========================================================= */

// app.get(
//   "/api/users/current",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       if (!req.userId) {
//         res.status(401).json({
//           success: false,
//           message: "Authenticated user ID was not found",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const userQueryConditions: Record<string, unknown>[] = [
//         {
//           id: req.userId,
//         },
//       ];

//       if (req.userEmail) {
//         userQueryConditions.push({
//           email: req.userEmail.toLowerCase(),
//         });
//       }

//       const currentUser = await usersCollection.findOne({
//         $or: userQueryConditions,
//       });

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const currentRole: UserRole =
//         currentUser.role === "admin" ||
//         currentUser.role === "doctor" ||
//         currentUser.role === "patient"
//           ? currentUser.role
//           : "patient";

//       const currentStatus: "active" | "blocked" =
//         currentUser.status === "blocked" ? "blocked" : "active";

//       const currentUserId =
//         typeof currentUser.id === "string" && currentUser.id.trim()
//           ? currentUser.id
//           : currentUser._id instanceof ObjectId
//             ? currentUser._id.toHexString()
//             : req.userId;

//       res.status(200).json({
//         id: currentUserId,
//         _id: currentUserId,
//         name: typeof currentUser.name === "string" ? currentUser.name : null,
//         email: typeof currentUser.email === "string" ? currentUser.email : null,
//         image: typeof currentUser.image === "string" ? currentUser.image : null,
//         role: currentRole,
//         status: currentStatus,
//       });
//     } catch (error) {
//       console.error(
//         "Get current user error:",
//         error instanceof Error ? error.message : error,
//       );

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve current user",
//       });
//     }
//   },
// );

// /* =========================================================
//    Manage Doctors helpers
// ========================================================= */

// type DoctorStatus = "active" | "blocked";

// const getDoctorString = (value: unknown): string => {
//   return typeof value === "string" ? value.trim() : "";
// };

// const getDoctorNumber = (value: unknown): number => {
//   const numberValue =
//     typeof value === "number"
//       ? value
//       : typeof value === "string" && value.trim()
//         ? Number(value)
//         : 0;

//   return Number.isFinite(numberValue) ? Math.max(0, numberValue) : 0;
// };

// const normalizeDoctorEmail = (value: unknown): string => {
//   return getDoctorString(value).toLowerCase();
// };

// const isValidDoctorEmail = (email: string): boolean => {
//   return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
// };

// const escapeDoctorSearch = (value: string): string => {
//   return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// };

// const getDoctorDocumentId = (document: Document): string => {
//   if (typeof document.id === "string" && document.id.trim()) {
//     return document.id;
//   }

//   if (document._id instanceof ObjectId) {
//     return document._id.toHexString();
//   }

//   return String(document._id || "");
// };

// const getDoctorFilter = (doctorId: string): Filter<Document> => {
//   const conditions: Filter<Document>[] = [
//     {
//       id: doctorId,
//     },
//   ];

//   if (ObjectId.isValid(doctorId)) {
//     conditions.push({
//       _id: new ObjectId(doctorId),
//     });
//   }

//   return {
//     $or: conditions,
//   };
// };

// const getUserFilter = (userId: string): Filter<Document> => {
//   const conditions: Filter<Document>[] = [
//     {
//       id: userId,
//     },
//   ];

//   if (ObjectId.isValid(userId)) {
//     conditions.push({
//       _id: new ObjectId(userId),
//     });
//   }

//   return {
//     $or: conditions,
//   };
// };

// const formatDoctorDate = (value: unknown): string | null => {
//   if (value instanceof Date) {
//     return value.toISOString();
//   }

//   if (typeof value === "string" || typeof value === "number") {
//     const date = new Date(value);

//     return Number.isNaN(date.getTime()) ? null : date.toISOString();
//   }

//   return null;
// };

// const formatDoctor = (doctor: Document) => {
//   return {
//     id: getDoctorDocumentId(doctor),

//     userId: typeof doctor.userId === "string" ? doctor.userId : "",

//     name: getDoctorString(doctor.name),

//     email: normalizeDoctorEmail(doctor.email),

//     image: getDoctorString(doctor.image) || null,

//     phone: getDoctorString(doctor.phone),

//     specialization: getDoctorString(doctor.specialization),

//     qualification: getDoctorString(doctor.qualification),

//     experienceYears: getDoctorNumber(doctor.experienceYears),

//     hospital:
//       getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),

//     address: getDoctorString(doctor.address),

//     bio: getDoctorString(doctor.bio),

//     role: "doctor" as const,

//     status:
//       doctor.status === "blocked" ? ("blocked" as const) : ("active" as const),

//     createdAt: formatDoctorDate(doctor.createdAt),

//     updatedAt: formatDoctorDate(doctor.updatedAt),
//   };
// };

// const readBetterAuthResponse = async (
//   response: globalThis.Response,
// ): Promise<unknown> => {
//   try {
//     return await response.json();
//   } catch {
//     return null;
//   }
// };

// const getBetterAuthError = (value: unknown): string => {
//   if (typeof value !== "object" || value === null) {
//     return "Doctor authentication account could not be created";
//   }

//   const data = value as Record<string, unknown>;

//   if (typeof data.message === "string" && data.message.trim()) {
//     return data.message;
//   }

//   if (typeof data.error === "object" && data.error !== null) {
//     const error = data.error as Record<string, unknown>;

//     if (typeof error.message === "string" && error.message.trim()) {
//       return error.message;
//     }
//   }

//   return "Doctor authentication account could not be created";
// };

// /* =========================================================
//    Admin patient management helpers
// ========================================================= */

// type ManagedPatientStatus = "active" | "blocked";

// const getAdminPatientFilter = (patientId: string): Filter<Document> => {
//   return {
//     $and: [getUserFilter(patientId), { role: "patient" }],
//   };
// };

// const formatManagedPatient = (patient: Document) => {
//   const status: ManagedPatientStatus =
//     patient.status === "blocked" ? "blocked" : "active";

//   return {
//     id: getDoctorDocumentId(patient),
//     name: getDoctorString(patient.name),
//     email: normalizeDoctorEmail(patient.email),
//     image: getDoctorString(patient.image) || null,
//     role: "patient" as const,
//     status,
//     emailVerified: patient.emailVerified === true,
//     phone: getDoctorString(patient.phone) || null,
//     address: getDoctorString(patient.address) || null,
//     dateOfBirth: getDoctorString(patient.dateOfBirth) || null,
//     gender: getDoctorString(patient.gender) || null,
//     bloodGroup: getDoctorString(patient.bloodGroup) || null,
//     occupation: getDoctorString(patient.occupation) || null,
//     city: getDoctorString(patient.city) || null,
//     country: getDoctorString(patient.country) || null,
//     bio: getDoctorString(patient.bio) || null,
//     emergencyContactName: getDoctorString(patient.emergencyContactName) || null,
//     emergencyContactPhone:
//       getDoctorString(patient.emergencyContactPhone) ||
//       getDoctorString(patient.emergencyContact) ||
//       null,
//     createdAt: formatDoctorDate(patient.createdAt),
//     updatedAt: formatDoctorDate(patient.updatedAt),
//   };
// };

// /* =========================================================
//    GET managed patients (10 per page)
// ========================================================= */

// app.get(
//   "/api/v1/admin/patients",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const search = getDoctorString(req.query.search);
//       const requestedStatus = getDoctorString(req.query.status);
//       const requestedPage = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;

//       const conditions: Filter<Document>[] = [{ role: "patient" }];

//       if (requestedStatus === "active" || requestedStatus === "blocked") {
//         conditions.push({ status: requestedStatus });
//       }

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         conditions.push({
//           $or: [
//             {
//               name: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               email: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       const filter: Filter<Document> = { $and: conditions };
//       const usersCollection = database.collection("user");
//       const total = await usersCollection.countDocuments(filter);
//       const totalPages = Math.max(1, Math.ceil(total / limit));
//       const page = Math.min(requestedPage, totalPages);

//       const patientDocuments = await usersCollection
//         .find(filter)
//         .sort({
//           updatedAt: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .skip((page - 1) * limit)
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         patients: patientDocuments.map(formatManagedPatient),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages,
//         },
//       });
//     } catch (error) {
//       console.error("Get managed patients error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patients",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET managed patient details
// ========================================================= */

// app.get(
//   "/api/v1/admin/patients/:patientId",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);

//       if (!patientId) {
//         res.status(400).json({
//           success: false,
//           message: "Patient ID is required",
//         });
//         return;
//       }

//       const patient = await database
//         .collection("user")
//         .findOne(getAdminPatientFilter(patientId));

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         patient: formatManagedPatient(patient),
//       });
//     } catch (error) {
//       console.error("Get managed patient details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient details",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH block or activate patient
// ========================================================= */

// app.patch(
//   "/api/v1/admin/patients/:patientId/status",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);
//       const requestedStatus = getDoctorString(req.body.status);

//       if (requestedStatus !== "active" && requestedStatus !== "blocked") {
//         res.status(400).json({
//           success: false,
//           message: "Status must be active or blocked",
//         });
//         return;
//       }

//       const usersCollection = database.collection("user");
//       const patient = await usersCollection.findOne(
//         getAdminPatientFilter(patientId),
//       );

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       const status = requestedStatus as ManagedPatientStatus;
//       const updatedPatient = await usersCollection.findOneAndUpdate(
//         { _id: patient._id },
//         {
//           $set: {
//             status,
//             updatedAt: new Date(),
//           },
//         },
//         { returnDocument: "after" },
//       );

//       if (!updatedPatient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       if (status === "blocked") {
//         await database.collection("session").deleteMany({
//           userId: getDoctorDocumentId(patient),
//         });
//       }

//       res.status(200).json({
//         success: true,
//         message:
//           status === "blocked"
//             ? "Patient blocked successfully"
//             : "Patient activated successfully",
//         patient: formatManagedPatient(updatedPatient),
//       });
//     } catch (error) {
//       console.error("Change patient status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to change patient status",
//       });
//     }
//   },
// );

// /* =========================================================
//    DELETE patient account
// ========================================================= */

// app.delete(
//   "/api/v1/admin/patients/:patientId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);

//       if (!patientId) {
//         res.status(400).json({
//           success: false,
//           message: "Patient ID is required",
//         });
//         return;
//       }

//       const usersCollection = database.collection("user");
//       const patient = await usersCollection.findOne(
//         getAdminPatientFilter(patientId),
//       );

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(patient);
//       const email = normalizeDoctorEmail(patient.email);

//       await Promise.all([
//         database.collection("session").deleteMany({ userId }),
//         database.collection("account").deleteMany({ userId }),
//         database.collection("verification").deleteMany({
//           $or: [{ identifier: email }, { value: email }],
//         }),
//       ]);

//       const deleteResult = await usersCollection.deleteOne({
//         _id: patient._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Patient account could not be deleted",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Patient account deleted successfully",
//         deletedPatientId: userId,
//       });
//     } catch (error) {
//       console.error("Delete patient account error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete patient account",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET all doctors
// ========================================================= */

// app.get(
//   "/api/v1/admin/doctors",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const search = getDoctorString(req.query.search);

//       const status = getDoctorString(req.query.status);

//       const page = Math.max(
//         1,
//         Math.floor(getDoctorNumber(req.query.page) || 1),
//       );

//       const limit = Math.min(
//         100,
//         Math.max(1, Math.floor(getDoctorNumber(req.query.limit) || 50)),
//       );

//       const filter: Filter<Document> = {};

//       if (status === "active" || status === "blocked") {
//         filter.status = status;
//       }

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         filter.$or = [
//           {
//             name: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             email: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             phone: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             specialization: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//         ];
//       }

//       const [doctorDocuments, total] = await Promise.all([
//         doctorsCollection
//           .find(filter)
//           .sort({
//             createdAt: -1,
//             _id: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         doctorsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,

//         doctors: doctorDocuments.map(formatDoctor),

//         pagination: {
//           page,
//           limit,
//           total,

//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get doctors error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET single doctor details
// ========================================================= */

// app.get(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       if (!doctorId) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         doctor: formatDoctor(doctor),
//       });
//     } catch (error) {
//       console.error("Get doctor details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor details",
//       });
//     }
//   },
// );

// /* =========================================================
//    POST create doctor
// ========================================================= */

// app.post(
//   "/api/v1/admin/doctors",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const name = getDoctorString(req.body.name);

//       const email = normalizeDoctorEmail(req.body.email);

//       const password = getDoctorString(req.body.password);

//       const specialization = getDoctorString(req.body.specialization);

//       if (!name || !email || !password || !specialization) {
//         res.status(400).json({
//           success: false,
//           message: "Name, email, password and specialization are required",
//         });

//         return;
//       }

//       if (!isValidDoctorEmail(email)) {
//         res.status(400).json({
//           success: false,
//           message: "A valid email address is required",
//         });

//         return;
//       }

//       if (password.length < 8) {
//         res.status(400).json({
//           success: false,
//           message: "Password must contain at least 8 characters",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const doctorsCollection = database.collection("doctors");

//       const accountsCollection = database.collection("account");

//       const sessionsCollection = database.collection("session");

//       const existingUser = await usersCollection.findOne({
//         email,
//       });

//       const existingDoctor = await doctorsCollection.findOne({
//         email,
//       });

//       if (existingUser || existingDoctor) {
//         res.status(409).json({
//           success: false,
//           message: "An account with this email already exists",
//         });

//         return;
//       }

//       /*
//        * Better Auth securely creates the email/password account.
//        * Raw password MongoDB-তে save হবে না।
//        */
//       const signupResponse = await fetch(
//         `${betterAuthUrl}/api/auth/sign-up/email`,
//         {
//           method: "POST",

//           headers: {
//             "content-type": "application/json",
//             accept: "application/json",
//             origin: betterAuthUrl,
//           },

//           body: JSON.stringify({
//             name,
//             email,
//             password,
//           }),
//         },
//       );

//       const signupData = await readBetterAuthResponse(signupResponse);

//       if (!signupResponse.ok) {
//         res
//           .status(signupResponse.status >= 500 ? 502 : signupResponse.status)
//           .json({
//             success: false,
//             message: getBetterAuthError(signupData),
//           });

//         return;
//       }

//       const createdUser = await usersCollection.findOne({
//         email,
//       });

//       if (!createdUser) {
//         res.status(500).json({
//           success: false,
//           message: "Authentication account was created but user was not found",
//         });

//         return;
//       }

//       const userId = getDoctorDocumentId(createdUser);

//       const now = new Date();

//       await usersCollection.updateOne(
//         {
//           _id: createdUser._id,
//         },
//         {
//           $set: {
//             name,
//             email,
//             role: "doctor",
//             status: "active",
//             updatedAt: now,
//           },
//         },
//       );

//       /*
//        * Admin-created doctor will sign in manually.
//        * Remove any session created by signup.
//        */
//       await sessionsCollection.deleteMany({
//         userId,
//       });

//       const doctorDocument = {
//         userId,

//         name,
//         email,

//         image: getDoctorString(req.body.image) || null,

//         phone: getDoctorString(req.body.phone),

//         specialization,

//         qualification: getDoctorString(req.body.qualification),

//         experienceYears: getDoctorNumber(req.body.experienceYears),

//         hospital: getDoctorString(req.body.hospital),

//         address: getDoctorString(req.body.address),

//         bio: getDoctorString(req.body.bio),

//         role: "doctor" as const,

//         status: "active" as const,

//         createdAt: now,
//         updatedAt: now,
//       };

//       try {
//         const insertResult = await doctorsCollection.insertOne(doctorDocument);

//         const createdDoctor = await doctorsCollection.findOne({
//           _id: insertResult.insertedId,
//         });

//         if (!createdDoctor) {
//           throw new Error("Created doctor profile was not found");
//         }

//         res.status(201).json({
//           success: true,
//           message: "Doctor created successfully",
//           doctor: formatDoctor(createdDoctor),
//         });
//       } catch (profileError) {
//         /*
//          * Roll back authentication data if
//          * doctor profile creation fails.
//          */
//         await Promise.all([
//           sessionsCollection.deleteMany({
//             userId,
//           }),

//           accountsCollection.deleteMany({
//             userId,
//           }),

//           usersCollection.deleteOne({
//             _id: createdUser._id,
//           }),
//         ]);

//         throw profileError;
//       }
//     } catch (error) {
//       console.error("Create doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to create doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH edit doctor
// ========================================================= */

// app.patch(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const name = getDoctorString(req.body.name);

//       const email = normalizeDoctorEmail(req.body.email);

//       const specialization = getDoctorString(req.body.specialization);

//       if (!doctorId || !name || !email || !specialization) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID, name, email and specialization are required",
//         });

//         return;
//       }

//       if (!isValidDoctorEmail(email)) {
//         res.status(400).json({
//           success: false,
//           message: "A valid email address is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       const linkedUser = userId
//         ? await usersCollection.findOne(getUserFilter(userId))
//         : null;

//       const duplicateDoctor = await doctorsCollection.findOne({
//         email,

//         _id: {
//           $ne: doctor._id,
//         },
//       });

//       const duplicateUser = await usersCollection.findOne({
//         email,

//         ...(linkedUser
//           ? {
//               _id: {
//                 $ne: linkedUser._id,
//               },
//             }
//           : {}),
//       });

//       if (duplicateDoctor || duplicateUser) {
//         res.status(409).json({
//           success: false,
//           message: "Another account already uses this email",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedDoctor = await doctorsCollection.findOneAndUpdate(
//         {
//           _id: doctor._id,
//         },
//         {
//           $set: {
//             name,
//             email,

//             image: getDoctorString(req.body.image) || null,

//             phone: getDoctorString(req.body.phone),

//             specialization,

//             qualification: getDoctorString(req.body.qualification),

//             experienceYears: getDoctorNumber(req.body.experienceYears),

//             hospital: getDoctorString(req.body.hospital),

//             address: getDoctorString(req.body.address),

//             bio: getDoctorString(req.body.bio),

//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       if (!updatedDoctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       if (linkedUser) {
//         await usersCollection.updateOne(
//           {
//             _id: linkedUser._id,
//           },
//           {
//             $set: {
//               name,
//               email,

//               image: getDoctorString(req.body.image) || null,

//               updatedAt: now,
//             },
//           },
//         );
//       }

//       res.status(200).json({
//         success: true,
//         message: "Doctor updated successfully",
//         doctor: formatDoctor(updatedDoctor),
//       });
//     } catch (error) {
//       console.error("Update doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH block or activate doctor
// ========================================================= */

// app.patch(
//   "/api/v1/admin/doctors/:doctorId/status",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const requestedStatus = getDoctorString(req.body.status);

//       if (requestedStatus !== "active" && requestedStatus !== "blocked") {
//         res.status(400).json({
//           success: false,
//           message: "Status must be active or blocked",
//         });

//         return;
//       }

//       const status = requestedStatus as DoctorStatus;

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedDoctor = await doctorsCollection.findOneAndUpdate(
//         {
//           _id: doctor._id,
//         },
//         {
//           $set: {
//             status,
//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       if (!updatedDoctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       if (userId) {
//         await usersCollection.updateOne(getUserFilter(userId), {
//           $set: {
//             status,
//             updatedAt: now,
//           },
//         });
//       }

//       res.status(200).json({
//         success: true,

//         message:
//           status === "blocked"
//             ? "Doctor blocked successfully"
//             : "Doctor activated successfully",

//         doctor: formatDoctor(updatedDoctor),
//       });
//     } catch (error) {
//       console.error("Change doctor status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to change doctor status",
//       });
//     }
//   },
// );

// /* =========================================================
//    DELETE doctor
// ========================================================= */

// app.delete(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       if (!doctorId) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const accountsCollection = database.collection("account");

//       const sessionsCollection = database.collection("session");

//       const verificationCollection = database.collection("verification");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       if (userId) {
//         await Promise.all([
//           sessionsCollection.deleteMany({
//             userId,
//           }),

//           accountsCollection.deleteMany({
//             userId,
//           }),

//           verificationCollection.deleteMany({
//             $or: [
//               {
//                 identifier: doctor.email,
//               },
//               {
//                 value: doctor.email,
//               },
//             ],
//           }),
//         ]);

//         await usersCollection.deleteOne(getUserFilter(userId));
//       }

//       const deleteResult = await doctorsCollection.deleteOne({
//         _id: doctor._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Doctor could not be deleted",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Doctor deleted successfully",
//         deletedDoctorId: getDoctorDocumentId(doctor),
//       });
//     } catch (error) {
//       console.error("Delete doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctors, appointments and reviews
// ========================================================= */

// type AppointmentStatus = "pending" | "approved" | "completed" | "rejected";

// const ACTIVE_APPOINTMENT_STATUSES: AppointmentStatus[] = [
//   "pending",
//   "approved",
// ];

// const getPositiveInteger = (
//   value: unknown,
//   fallback: number,
//   maximum: number,
// ): number => {
//   const parsed = Number(value);

//   if (!Number.isFinite(parsed)) {
//     return fallback;
//   }

//   return Math.min(maximum, Math.max(1, Math.floor(parsed)));
// };

// const getCurrentDatabaseUser = async (
//   req: AuthenticatedRequest,
// ): Promise<Document | null> => {
//   if (!database || !req.userId) {
//     return null;
//   }

//   const conditions: Filter<Document>[] = [getUserFilter(req.userId)];

//   if (req.userEmail) {
//     conditions.push({
//       email: req.userEmail.toLowerCase(),
//     });
//   }

//   return database.collection("user").findOne({
//     $or: conditions,
//   });
// };

// const getNormalizedUserRole = (user: Document): UserRole => {
//   return user.role === "admin" ||
//     user.role === "doctor" ||
//     user.role === "patient"
//     ? user.role
//     : "patient";
// };

// const getNormalizedUserStatus = (user: Document): UserStatus => {
//   return user.status === "blocked" ? "blocked" : "active";
// };

// const getPublicDoctor = (doctor: Document) => {
//   const ratingAverage = Number(doctor.ratingAverage);

//   const ratingCount = Number(doctor.ratingCount);

//   return {
//     ...formatDoctor(doctor),

//     ratingAverage: Number.isFinite(ratingAverage)
//       ? Number(ratingAverage.toFixed(1))
//       : 0,

//     ratingCount: Number.isFinite(ratingCount)
//       ? Math.max(0, Math.floor(ratingCount))
//       : 0,
//   };
// };

// const getReviewDocumentId = (document: Document): string => {
//   return getDoctorDocumentId(document);
// };

// const formatReview = (review: Document) => {
//   return {
//     id: getReviewDocumentId(review),
//     doctorId: getDoctorString(review.doctorId),
//     userId: getDoctorString(review.userId),
//     userName: getDoctorString(review.userName),
//     userEmail: normalizeDoctorEmail(review.userEmail),
//     userImage: getDoctorString(review.userImage) || null,
//     rating: Math.min(
//       5,
//       Math.max(1, Math.floor(getDoctorNumber(review.rating))),
//     ),
//     review: getDoctorString(review.review),
//     createdAt: formatDoctorDate(review.createdAt),
//     updatedAt: formatDoctorDate(review.updatedAt),
//   };
// };

// const refreshDoctorRatingStats = async (doctorId: string): Promise<void> => {
//   if (!database) {
//     return;
//   }

//   const reviewsCollection = database.collection("reviews");

//   const doctorsCollection = database.collection("doctors");

//   const [stats] = await reviewsCollection
//     .aggregate([
//       {
//         $match: {
//           doctorId,
//         },
//       },
//       {
//         $group: {
//           _id: "$doctorId",
//           ratingAverage: {
//             $avg: "$rating",
//           },
//           ratingCount: {
//             $sum: 1,
//           },
//         },
//       },
//     ])
//     .toArray();

//   await doctorsCollection.updateOne(getDoctorFilter(doctorId), {
//     $set: {
//       ratingAverage:
//         typeof stats?.ratingAverage === "number"
//           ? Number(stats.ratingAverage.toFixed(2))
//           : 0,
//       ratingCount:
//         typeof stats?.ratingCount === "number" ? stats.ratingCount : 0,
//       updatedAt: new Date(),
//     },
//   });
// };

// const formatAppointment = (appointment: Document) => {
//   return {
//     id: getDoctorDocumentId(appointment),
//     doctorId: getDoctorString(appointment.doctorId),
//     doctorUserId: getDoctorString(appointment.doctorUserId),
//     doctorName: getDoctorString(appointment.doctorName),
//     doctorImage: getDoctorString(appointment.doctorImage) || null,
//     specialization: getDoctorString(appointment.specialization),
//     hospital: getDoctorString(appointment.hospital),
//     patientUserId: getDoctorString(appointment.patientUserId),
//     patientName: getDoctorString(appointment.patientName),
//     patientEmail: normalizeDoctorEmail(appointment.patientEmail),
//     patientImage: getDoctorString(appointment.patientImage) || null,
//     phone: getDoctorString(appointment.phone),
//     address: getDoctorString(appointment.address),
//     problemTitle: getDoctorString(appointment.problemTitle),
//     symptomsDescription: getDoctorString(appointment.symptomsDescription),
//     appointmentDate: getDoctorString(appointment.appointmentDate),
//     appointmentTime: getDoctorString(appointment.appointmentTime),
//     status:
//       appointment.status === "approved" ||
//       appointment.status === "completed" ||
//       appointment.status === "rejected"
//         ? appointment.status
//         : "pending",
//     rejectionReason: getDoctorString(appointment.rejectionReason) || null,
//     approvedAt: formatDoctorDate(appointment.approvedAt),
//     completedAt: formatDoctorDate(appointment.completedAt),
//     rejectedAt: formatDoctorDate(appointment.rejectedAt),
//     rescheduledAt: formatDoctorDate(appointment.rescheduledAt),
//     rescheduledBy: getDoctorString(appointment.rescheduledBy) || null,
//     rescheduleReason: getDoctorString(appointment.rescheduleReason) || null,
//     createdAt: formatDoctorDate(appointment.createdAt),
//     updatedAt: formatDoctorDate(appointment.updatedAt),
//   };
// };

// const attachPatientImages = async (
//   appointments: Document[],
// ): Promise<Document[]> => {
//   if (!database || appointments.length === 0) {
//     return appointments;
//   }

//   const patientUserIds = Array.from(
//     new Set(
//       appointments
//         .map((appointment) => getDoctorString(appointment.patientUserId))
//         .filter(Boolean),
//     ),
//   );

//   if (patientUserIds.length === 0) {
//     return appointments;
//   }

//   const objectIds = patientUserIds
//     .filter((userId) => ObjectId.isValid(userId))
//     .map((userId) => new ObjectId(userId));

//   const userConditions: Filter<Document>[] = [
//     {
//       id: {
//         $in: patientUserIds,
//       },
//     },
//   ];

//   if (objectIds.length > 0) {
//     userConditions.push({
//       _id: {
//         $in: objectIds,
//       },
//     });
//   }

//   const users = await database
//     .collection("user")
//     .find(
//       {
//         $or: userConditions,
//       },
//       {
//         projection: {
//           id: 1,
//           image: 1,
//         },
//       },
//     )
//     .toArray();

//   const imageByUserId = new Map<string, string | null>();

//   users.forEach((user) => {
//     imageByUserId.set(
//       getDoctorDocumentId(user),
//       getDoctorString(user.image) || null,
//     );
//   });

//   return appointments.map((appointment) => ({
//     ...appointment,
//     patientImage:
//       getDoctorString(appointment.patientImage) ||
//       imageByUserId.get(getDoctorString(appointment.patientUserId)) ||
//       null,
//   }));
// };

// /* =========================================================
//    Public doctor filters
// ========================================================= */

// app.get(
//   "/api/v1/doctors/filters",
//   async (_req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorDocuments = await database
//         .collection("doctors")
//         .find(
//           {
//             status: "active",
//           },
//           {
//             projection: {
//               specialization: 1,
//               qualification: 1,
//               experienceYears: 1,
//               hospital: 1,
//               chamber: 1,
//             },
//           },
//         )
//         .toArray();

//       const specializations = new Set<string>();
//       const qualifications = new Set<string>();
//       const hospitals = new Set<string>();
//       const experienceYears = new Set<number>();

//       doctorDocuments.forEach((doctor) => {
//         const specialization = getDoctorString(doctor.specialization);
//         const qualification = getDoctorString(doctor.qualification);
//         const hospital =
//           getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber);
//         const experience = getDoctorNumber(doctor.experienceYears);

//         if (specialization) {
//           specializations.add(specialization);
//         }

//         if (qualification) {
//           qualifications.add(qualification);
//         }

//         if (hospital) {
//           hospitals.add(hospital);
//         }

//         experienceYears.add(experience);
//       });

//       res.status(200).json({
//         success: true,
//         filters: {
//           specializations: Array.from(specializations).sort((a, b) =>
//             a.localeCompare(b),
//           ),
//           qualifications: Array.from(qualifications).sort((a, b) =>
//             a.localeCompare(b),
//           ),
//           hospitals: Array.from(hospitals).sort((a, b) => a.localeCompare(b)),
//           experienceYears: Array.from(experienceYears).sort((a, b) => a - b),
//         },
//       });
//     } catch (error) {
//       console.error("Get public doctor filters error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor filters",
//       });
//     }
//   },
// );

// /* =========================================================
//    Top Rated Public Doctors
// ========================================================= */

// app.get(
//   "/api/v1/doctors/top-rated",
//   async (_req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const doctors = await doctorsCollection
//         .find({
//           status: "active",
//         })
//         .sort({
//           ratingAverage: -1,
//           ratingCount: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .limit(4)
//         .toArray();

//       res.status(200).json({
//         success: true,

//         doctors: doctors.map(getPublicDoctor),
//       });
//     } catch (error) {
//       console.error("Get top rated doctors error:", error);

//       res.status(500).json({
//         success: false,

//         message: "Failed to retrieve top rated doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctor list
// ========================================================= */

// app.get(
//   "/api/v1/doctors",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const search = getDoctorString(req.query.search);
//       const specialization = getDoctorString(req.query.specialization);
//       const qualification = getDoctorString(req.query.qualification);
//       const hospital = getDoctorString(req.query.hospital);
//       const experienceValue = getDoctorString(req.query.experienceYears);

//       const page = getPositiveInteger(req.query.page, 1, 100000);

//       const limit = 8;

//       const conditions: Filter<Document>[] = [
//         {
//           status: "active",
//         },
//       ];

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         conditions.push({
//           $or: [
//             {
//               name: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               specialization: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               qualification: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       if (specialization) {
//         conditions.push({
//           specialization: {
//             $regex: `^${escapeDoctorSearch(specialization)}$`,
//             $options: "i",
//           },
//         });
//       }

//       if (qualification) {
//         conditions.push({
//           qualification: {
//             $regex: `^${escapeDoctorSearch(qualification)}$`,
//             $options: "i",
//           },
//         });
//       }

//       if (hospital) {
//         const safeHospital = `^${escapeDoctorSearch(hospital)}$`;

//         conditions.push({
//           $or: [
//             {
//               hospital: {
//                 $regex: safeHospital,
//                 $options: "i",
//               },
//             },
//             {
//               chamber: {
//                 $regex: safeHospital,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       if (experienceValue) {
//         const experienceYears = Number(experienceValue);

//         if (Number.isFinite(experienceYears)) {
//           conditions.push({
//             experienceYears: Math.max(0, Math.floor(experienceYears)),
//           });
//         }
//       }

//       const filter: Filter<Document> = {
//         $and: conditions,
//       };

//       const doctorsCollection = database.collection("doctors");

//       const [doctorDocuments, total] = await Promise.all([
//         doctorsCollection
//           .find(filter)
//           .sort({
//             ratingAverage: -1,
//             ratingCount: -1,
//             createdAt: -1,
//             _id: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         doctorsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         doctors: doctorDocuments.map(getPublicDoctor),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get public doctors error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve public doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public single doctor details
// ========================================================= */

// app.get(
//   "/api/v1/doctors/:doctorId",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         doctor: getPublicDoctor(doctor),
//       });
//     } catch (error) {
//       console.error("Get public doctor details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctor reviews
// ========================================================= */

// app.get(
//   "/api/v1/doctors/:doctorId/reviews",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = getPositiveInteger(req.query.limit, 10, 50);

//       const reviewsCollection = database.collection("reviews");

//       const [reviewDocuments, total] = await Promise.all([
//         reviewsCollection
//           .find({
//             doctorId,
//           })
//           .sort({
//             updatedAt: -1,
//             createdAt: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         reviewsCollection.countDocuments({
//           doctorId,
//         }),
//       ]);

//       res.status(200).json({
//         success: true,
//         reviews: reviewDocuments.map(formatReview),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get doctor reviews error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor reviews",
//       });
//     }
//   },
// );

// /* =========================================================
//    Create doctor review
// ========================================================= */

// app.post(
//   "/api/v1/doctors/:doctorId/reviews",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot submit a rating or review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const rating = Math.floor(Number(req.body.rating));
//       const reviewText = getDoctorString(req.body.review);

//       if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
//         res.status(400).json({
//           success: false,
//           message: "Rating must be a number from 1 to 5",
//         });

//         return;
//       }

//       if (reviewText.length > 2000) {
//         res.status(400).json({
//           success: false,
//           message: "Review cannot contain more than 2000 characters",
//         });

//         return;
//       }

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(doctor.userId) === currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "A doctor cannot review their own profile",
//         });

//         return;
//       }

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         doctorId,
//         userId: currentUserId,
//       });

//       if (existingReview) {
//         res.status(409).json({
//           success: false,
//           message:
//             "You have already reviewed this doctor. Please edit your existing review.",
//           code: "REVIEW_ALREADY_EXISTS",
//         });

//         return;
//       }

//       const now = new Date();

//       const reviewDocument = {
//         doctorId,
//         doctorUserId: getDoctorString(doctor.userId),
//         userId: currentUserId,
//         userName: getDoctorString(currentUser.name),
//         userEmail: normalizeDoctorEmail(currentUser.email),
//         userImage: getDoctorString(currentUser.image) || null,
//         rating,
//         review: reviewText,
//         createdAt: now,
//         updatedAt: now,
//       };

//       const insertResult = await reviewsCollection.insertOne(reviewDocument);

//       await refreshDoctorRatingStats(doctorId);

//       const createdReview = await reviewsCollection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "Rating and review submitted successfully",
//         review: createdReview ? formatReview(createdReview) : null,
//       });
//     } catch (error) {
//       console.error("Create doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to submit rating and review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Update doctor review
// ========================================================= */

// app.patch(
//   "/api/v1/doctors/:doctorId/reviews/:reviewId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot edit a review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const reviewId = getDoctorString(req.params.reviewId);
//       const rating = Math.floor(Number(req.body.rating));
//       const reviewText = getDoctorString(req.body.review);

//       if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
//         res.status(400).json({
//           success: false,
//           message: "Rating must be a number from 1 to 5",
//         });

//         return;
//       }

//       if (reviewText.length > 2000) {
//         res.status(400).json({
//           success: false,
//           message: "Review cannot contain more than 2000 characters",
//         });

//         return;
//       }

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         $and: [
//           getDoctorFilter(reviewId),
//           {
//             doctorId,
//           },
//         ],
//       });

//       if (!existingReview) {
//         res.status(404).json({
//           success: false,
//           message: "Review was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(existingReview.userId) !== currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can edit only your own review",
//         });

//         return;
//       }

//       const updatedReview = await reviewsCollection.findOneAndUpdate(
//         {
//           _id: existingReview._id,
//         },
//         {
//           $set: {
//             rating,
//             review: reviewText,
//             updatedAt: new Date(),
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       await refreshDoctorRatingStats(doctorId);

//       res.status(200).json({
//         success: true,
//         message: "Rating and review updated successfully",
//         review: updatedReview ? formatReview(updatedReview) : null,
//       });
//     } catch (error) {
//       console.error("Update doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update rating and review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Delete doctor review
// ========================================================= */

// app.delete(
//   "/api/v1/doctors/:doctorId/reviews/:reviewId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot delete a review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const reviewId = getDoctorString(req.params.reviewId);

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         $and: [
//           getDoctorFilter(reviewId),
//           {
//             doctorId,
//           },
//         ],
//       });

//       if (!existingReview) {
//         res.status(404).json({
//           success: false,
//           message: "Review was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(existingReview.userId) !== currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can delete only your own review",
//         });

//         return;
//       }

//       await reviewsCollection.deleteOne({
//         _id: existingReview._id,
//       });

//       await refreshDoctorRatingStats(doctorId);

//       res.status(200).json({
//         success: true,
//         message: "Review deleted successfully",
//         deletedReviewId: reviewId,
//       });
//     } catch (error) {
//       console.error("Delete doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Appointment eligibility
// ========================================================= */

// app.get(
//   "/api/v1/appointments/eligibility/:doctorId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);

//       if (role !== "patient") {
//         res.status(403).json({
//           success: false,
//           canBook: false,
//           code: "PATIENT_ONLY",
//           message: "Only patients can take a doctor appointment.",
//         });

//         return;
//       }

//       if (status === "blocked") {
//         res.status(403).json({
//           success: false,
//           canBook: false,
//           code: "ACCOUNT_BLOCKED",
//           message:
//             "You are restricted by the administrator and cannot take an appointment.",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           canBook: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);

//       const existingAppointment = await database
//         .collection("appointments")
//         .findOne({
//           doctorId,
//           patientUserId,
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//       if (existingAppointment) {
//         res.status(200).json({
//           success: true,
//           canBook: false,
//           code: "APPOINTMENT_ALREADY_EXISTS",
//           message:
//             "You already have a pending or approved appointment with this doctor.",
//           appointment: formatAppointment(existingAppointment),
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         canBook: true,
//         message: "You can take an appointment with this doctor.",
//       });
//     } catch (error) {
//       console.error("Appointment eligibility error:", error);

//       res.status(500).json({
//         success: false,
//         canBook: false,
//         message: "Failed to check appointment eligibility",
//       });
//     }
//   },
// );

// /* =========================================================
//    Create appointment
// ========================================================= */

// app.post(
//   "/api/v1/appointments",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);

//       if (role !== "patient") {
//         res.status(403).json({
//           success: false,
//           message: "Only patients can take a doctor appointment.",
//           code: "PATIENT_ONLY",
//         });

//         return;
//       }

//       if (status === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot take an appointment.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.body.doctorId);
//       const patientName = getDoctorString(req.body.patientName);
//       const phone = getDoctorString(req.body.phone);
//       const address = getDoctorString(req.body.address);
//       const problemTitle = getDoctorString(req.body.problemTitle);
//       const symptomsDescription = getDoctorString(req.body.symptomsDescription);
//       const appointmentDate = getDoctorString(req.body.appointmentDate);
//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       if (
//         !doctorId ||
//         !patientName ||
//         !phone ||
//         !address ||
//         !problemTitle ||
//         !symptomsDescription ||
//         !appointmentDate ||
//         !appointmentTime
//       ) {
//         res.status(400).json({
//           success: false,
//           message:
//             "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
//         });

//         return;
//       }

//       if (symptomsDescription.length > 5000) {
//         res.status(400).json({
//           success: false,
//           message:
//             "Symptoms description cannot contain more than 5000 characters",
//         });

//         return;
//       }

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentsCollection = database.collection("appointments");

//       const existingAppointment = await appointmentsCollection.findOne({
//         doctorId,
//         patientUserId,
//         status: {
//           $in: ACTIVE_APPOINTMENT_STATUSES,
//         },
//       });

//       if (existingAppointment) {
//         res.status(409).json({
//           success: false,
//           message:
//             "You already have a pending or approved appointment with this doctor.",
//           code: "APPOINTMENT_ALREADY_EXISTS",
//           appointment: formatAppointment(existingAppointment),
//         });

//         return;
//       }

//       const now = new Date();

//       const appointmentDocument = {
//         doctorId,
//         doctorUserId: getDoctorString(doctor.userId),
//         doctorName: getDoctorString(doctor.name),
//         doctorImage: getDoctorString(doctor.image) || null,
//         specialization: getDoctorString(doctor.specialization),
//         hospital:
//           getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
//         patientUserId,
//         patientName,
//         patientEmail: normalizeDoctorEmail(currentUser.email),
//         patientImage: getDoctorString(currentUser.image) || null,
//         phone,
//         address,
//         problemTitle,
//         symptomsDescription,
//         appointmentDate,
//         appointmentTime,
//         status: "pending" as const,
//         createdAt: now,
//         updatedAt: now,
//       };

//       const insertResult =
//         await appointmentsCollection.insertOne(appointmentDocument);

//       const createdAppointment = await appointmentsCollection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "Appointment request submitted successfully",
//         appointment: createdAppointment
//           ? formatAppointment(createdAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Create appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to submit appointment request",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient appointment helpers
// ========================================================= */

// const getPatientAppointment = async (
//   patientUserId: string,
//   appointmentId: string,
// ): Promise<Document | null> => {
//   if (!database) {
//     return null;
//   }

//   return database.collection("appointments").findOne({
//     $and: [
//       getDoctorFilter(appointmentId),
//       {
//         patientUserId,
//       },
//     ],
//   });
// };

// const getAppointmentDoctor = async (
//   appointment: Document,
// ): Promise<Document | null> => {
//   if (!database) {
//     return null;
//   }

//   const doctorId = getDoctorString(appointment.doctorId);

//   if (!doctorId) {
//     return null;
//   }

//   return database.collection("doctors").findOne(getDoctorFilter(doctorId));
// };

// const validatePatientAppointmentInput = (
//   body: Record<string, unknown>,
// ):
//   | {
//       success: true;
//       values: {
//         patientName: string;
//         phone: string;
//         address: string;
//         problemTitle: string;
//         symptomsDescription: string;
//         appointmentDate: string;
//         appointmentTime: string;
//       };
//     }
//   | {
//       success: false;
//       message: string;
//     } => {
//   const patientName = getDoctorString(body.patientName);
//   const phone = getDoctorString(body.phone);
//   const address = getDoctorString(body.address);
//   const problemTitle = getDoctorString(body.problemTitle);
//   const symptomsDescription = getDoctorString(body.symptomsDescription);
//   const appointmentDate = getDoctorString(body.appointmentDate);
//   const appointmentTime = getDoctorString(body.appointmentTime);

//   if (
//     !patientName ||
//     !phone ||
//     !address ||
//     !problemTitle ||
//     !symptomsDescription ||
//     !appointmentDate ||
//     !appointmentTime
//   ) {
//     return {
//       success: false,
//       message:
//         "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
//     };
//   }

//   if (patientName.length > 150) {
//     return {
//       success: false,
//       message: "Patient name cannot contain more than 150 characters",
//     };
//   }

//   if (phone.length > 40) {
//     return {
//       success: false,
//       message: "Phone number cannot contain more than 40 characters",
//     };
//   }

//   if (address.length > 500) {
//     return {
//       success: false,
//       message: "Address cannot contain more than 500 characters",
//     };
//   }

//   if (problemTitle.length > 250) {
//     return {
//       success: false,
//       message: "Health problem title cannot contain more than 250 characters",
//     };
//   }

//   if (symptomsDescription.length > 5000) {
//     return {
//       success: false,
//       message: "Symptoms description cannot contain more than 5000 characters",
//     };
//   }

//   const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//   const timePattern = /^\d{2}:\d{2}$/;

//   if (
//     !datePattern.test(appointmentDate) ||
//     !timePattern.test(appointmentTime)
//   ) {
//     return {
//       success: false,
//       message: "A valid appointment date and time are required",
//     };
//   }

//   const today = new Date().toISOString().slice(0, 10);

//   if (appointmentDate < today) {
//     return {
//       success: false,
//       message: "Appointment date cannot be in the past",
//     };
//   }

//   return {
//     success: true,
//     values: {
//       patientName,
//       phone,
//       address,
//       problemTitle,
//       symptomsDescription,
//       appointmentDate,
//       appointmentTime,
//     },
//   };
// };

// /* =========================================================
//    Patient appointments list
// ========================================================= */

// app.get(
//   "/api/v1/patient/appointments",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;
//       const appointmentsCollection = database.collection("appointments");
//       const filter: Filter<Document> = { patientUserId };

//       const [appointmentDocuments, total] = await Promise.all([
//         appointmentsCollection
//           .find(filter)
//           .sort({ createdAt: -1, _id: -1 })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),
//         appointmentsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         appointments: appointmentDocuments.map(formatAppointment),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get patient appointments error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient single appointment details
// ========================================================= */

// app.get(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);

//       if (!appointmentId) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment ID is required",
//         });
//         return;
//       }

//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       const doctor = await getAppointmentDoctor(appointment);

//       res.status(200).json({
//         success: true,
//         appointment: formatAppointment(appointment),
//         doctor: doctor ? getPublicDoctor(doctor) : null,
//       });
//     } catch (error) {
//       console.error("Get patient appointment details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointment details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient edit appointment
// ========================================================= */

// app.patch(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus !== "pending" && currentStatus !== "rejected") {
//         res.status(409).json({
//           success: false,
//           message: "Only pending or rejected appointments can be edited",
//         });
//         return;
//       }

//       const validation = validatePatientAppointmentInput(
//         req.body as Record<string, unknown>,
//       );

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       if (currentStatus === "rejected") {
//         const anotherActiveAppointment = await database
//           .collection("appointments")
//           .findOne({
//             _id: { $ne: appointment._id },
//             doctorId: getDoctorString(appointment.doctorId),
//             patientUserId,
//             status: { $in: ACTIVE_APPOINTMENT_STATUSES },
//           });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "You already have another pending or approved appointment with this doctor.",
//           });
//           return;
//         }
//       }

//       const now = new Date();
//       const updatedAppointment = await database
//         .collection("appointments")
//         .findOneAndUpdate(
//           { _id: appointment._id },
//           {
//             $set: {
//               ...validation.values,
//               status: "pending",
//               rejectionReason: null,
//               rejectedAt: null,
//               approvedAt: null,
//               completedAt: null,
//               updatedAt: now,
//             },
//           },
//           { returnDocument: "after" },
//         );

//       res.status(200).json({
//         success: true,
//         message:
//           currentStatus === "rejected"
//             ? "Appointment updated and resubmitted successfully"
//             : "Appointment updated successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update patient appointment error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient cancel and delete appointment
// ========================================================= */

// app.delete(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       if (getDoctorString(appointment.status) === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be cancelled or deleted",
//         });
//         return;
//       }

//       const deleteResult = await database
//         .collection("appointments")
//         .deleteOne({ _id: appointment._id });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Appointment could not be cancelled",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Appointment cancelled and removed successfully",
//         deletedAppointmentId: appointmentId,
//       });
//     } catch (error) {
//       console.error("Cancel patient appointment error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to cancel appointment",
//       });
//     }
//   },
// );

// const getAppointmentListFilter = (
//   req: AuthenticatedRequest,
// ): Filter<Document> => {
//   const conditions: Filter<Document>[] = [];
//   const status = getDoctorString(req.query.status);
//   const search = getDoctorString(req.query.search);

//   if (
//     status === "pending" ||
//     status === "approved" ||
//     status === "completed" ||
//     status === "rejected"
//   ) {
//     conditions.push({
//       status,
//     });
//   }

//   if (search) {
//     const safeSearch = escapeDoctorSearch(search);

//     conditions.push({
//       $or: [
//         {
//           patientName: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           patientEmail: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           doctorName: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           problemTitle: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//       ],
//     });
//   }

//   return conditions.length
//     ? {
//         $and: conditions,
//       }
//     : {};
// };

// const sendAppointmentList = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   additionalFilter: Filter<Document> = {},
// ): Promise<void> => {
//   if (!database) {
//     res.status(503).json({
//       success: false,
//       message: "Database is not connected",
//     });

//     return;
//   }

//   const page = getPositiveInteger(req.query.page, 1, 100000);
//   const limit = 10;

//   const queryFilter = getAppointmentListFilter(req);

//   const filter: Filter<Document> = {
//     $and: [queryFilter, additionalFilter],
//   };

//   const appointmentsCollection = database.collection("appointments");

//   const [appointmentDocuments, total] = await Promise.all([
//     appointmentsCollection
//       .find(filter)
//       .sort({
//         createdAt: -1,
//         _id: -1,
//       })
//       .skip((page - 1) * limit)
//       .limit(limit)
//       .toArray(),

//     appointmentsCollection.countDocuments(filter),
//   ]);

//   const appointmentsWithImages =
//     await attachPatientImages(appointmentDocuments);

//   res.status(200).json({
//     success: true,
//     appointments: appointmentsWithImages.map(formatAppointment),
//     pagination: {
//       page,
//       limit,
//       total,
//       totalPages: Math.max(1, Math.ceil(total / limit)),
//     },
//   });
// };

// /* =========================================================
//    Admin appointment management
// ========================================================= */

// app.get(
//   "/api/v1/admin/appointments",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       await sendAppointmentList(req, res);
//     } catch (error) {
//       console.error("Get admin appointments error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Doctor appointment management
// ========================================================= */

// app.get(
//   "/api/v1/doctor/appointments",
//   verifyToken,
//   verifyDoctor,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       await sendAppointmentList(req, res, {
//         doctorUserId,
//       });
//     } catch (error) {
//       console.error("Get doctor appointments error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Doctor single appointment details
// ========================================================= */

// app.get(
//   "/api/v1/doctor/appointments/:appointmentId",
//   verifyToken,
//   verifyDoctor,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       if (!appointmentId) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment ID is required",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       const appointment = await database.collection("appointments").findOne({
//         $and: [
//           getDoctorFilter(appointmentId),
//           {
//             doctorUserId,
//           },
//         ],
//       });

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const [appointmentWithImage] = await attachPatientImages([appointment]);

//       res.status(200).json({
//         success: true,
//         appointment: formatAppointment(appointmentWithImage),
//       });
//     } catch (error) {
//       console.error("Get doctor appointment details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointment details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// /* =========================================================
//    Doctor appointment reschedule
// ========================================================= */

// app.patch(
//   "/api/v1/doctor/appointments/:appointmentId/reschedule",
//   verifyToken,
//   verifyDoctor,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentDate = getDoctorString(req.body.appointmentDate);

//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       const rescheduleReason = getDoctorString(req.body.rescheduleReason);

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       if (rescheduleReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Reschedule reason cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(appointment.doctorUserId) !== doctorUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can reschedule only your own appointments",
//         });

//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (
//         currentStatus !== "pending" &&
//         currentStatus !== "approved" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message:
//             "Only pending, approved or rejected appointments can be rescheduled",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: {
//             appointmentDate,
//             appointmentTime,
//             rescheduleReason: rescheduleReason || null,
//             rescheduledAt: now,
//             rescheduledBy: doctorUserId,
//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message: "Appointment rescheduled successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Doctor reschedule appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to reschedule appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin-only appointment reschedule
// ========================================================= */

// app.patch(
//   "/api/v1/admin/appointments/:appointmentId/reschedule",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentDate = getDoctorString(req.body.appointmentDate);

//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       const rescheduleReason = getDoctorString(req.body.rescheduleReason);

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       if (rescheduleReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Reschedule reason cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed" || currentStatus === "rejected") {
//         res.status(409).json({
//           success: false,
//           message: "A completed or rejected appointment cannot be rescheduled",
//         });

//         return;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: {
//             appointmentDate,
//             appointmentTime,
//             rescheduleReason: rescheduleReason || null,
//             rescheduledAt: new Date(),
//             rescheduledBy: req.userId || null,
//             updatedAt: new Date(),
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message: "Appointment rescheduled successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Reschedule appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to reschedule appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// app.patch(
//   "/api/v1/appointments/:appointmentId/status",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const userStatus = getNormalizedUserStatus(currentUser);

//       if (role !== "admin" && role !== "doctor") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Only an administrator or doctor can update appointment status.",
//         });

//         return;
//       }

//       if (userStatus === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Your account is blocked. You can view appointments but cannot update them.",
//           code: "READ_ONLY_ACCOUNT",
//         });

//         return;
//       }

//       const requestedStatus = getDoctorString(req.body.status);
//       const rejectionReason = getDoctorString(req.body.rejectionReason);

//       if (
//         requestedStatus !== "approved" &&
//         requestedStatus !== "completed" &&
//         requestedStatus !== "rejected"
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "Status must be approved, completed or rejected",
//         });

//         return;
//       }

//       if (requestedStatus === "rejected" && !rejectionReason) {
//         res.status(400).json({
//           success: false,
//           message:
//             "A rejection message is required when rejecting an appointment",
//         });

//         return;
//       }

//       if (rejectionReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Rejection message cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       if (role === "doctor") {
//         const currentUserId = getDoctorDocumentId(currentUser);

//         if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
//           res.status(403).json({
//             success: false,
//             message: "You can update only your own appointments",
//           });

//           return;
//         }
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be changed",
//         });

//         return;
//       }

//       if (requestedStatus === "completed" && currentStatus !== "approved") {
//         res.status(409).json({
//           success: false,
//           message: "Only an approved appointment can be marked as completed",
//         });

//         return;
//       }

//       if (
//         requestedStatus === "approved" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or rejected appointment can be approved",
//         });

//         return;
//       }

//       if (
//         requestedStatus === "rejected" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "approved"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or approved appointment can be rejected",
//         });

//         return;
//       }

//       if (requestedStatus === "approved" && currentStatus === "rejected") {
//         const anotherActiveAppointment = await appointmentsCollection.findOne({
//           _id: {
//             $ne: appointment._id,
//           },
//           doctorId: getDoctorString(appointment.doctorId),
//           patientUserId: getDoctorString(appointment.patientUserId),
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "This patient already has another pending or approved appointment with you.",
//           });

//           return;
//         }
//       }

//       const now = new Date();

//       const statusFields: Record<string, unknown> = {
//         status: requestedStatus as AppointmentStatus,
//         rejectionReason:
//           requestedStatus === "rejected" ? rejectionReason : null,
//         updatedAt: now,
//       };

//       if (requestedStatus === "approved") {
//         statusFields.approvedAt = now;
//         statusFields.rejectedAt = null;
//         statusFields.rejectionReason = null;
//       }

//       if (requestedStatus === "completed") {
//         statusFields.completedAt = now;
//       }

//       if (requestedStatus === "rejected") {
//         statusFields.rejectedAt = now;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: statusFields,
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message:
//           requestedStatus === "approved"
//             ? "Appointment approved successfully"
//             : requestedStatus === "completed"
//               ? "Consultation completed successfully."
//               : "Appointment rejected successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update appointment status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment status",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// app.patch(
//   "/api/v1/appointments/:appointmentId/status",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const userStatus = getNormalizedUserStatus(currentUser);

//       if (role !== "admin" && role !== "doctor") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Only an administrator or doctor can update appointment status.",
//         });
//         return;
//       }

//       if (userStatus === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Your account is blocked. You can view appointments but cannot update them.",
//           code: "READ_ONLY_ACCOUNT",
//         });
//         return;
//       }

//       const requestedStatus = getDoctorString(req.body.status);
//       const rejectionReason = getDoctorString(req.body.rejectionReason);

//       if (
//         requestedStatus !== "approved" &&
//         requestedStatus !== "completed" &&
//         requestedStatus !== "rejected"
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "Status must be approved, completed or rejected",
//         });
//         return;
//       }

//       if (requestedStatus === "rejected" && !rejectionReason) {
//         res.status(400).json({
//           success: false,
//           message:
//             "A rejection message is required when rejecting an appointment",
//         });
//         return;
//       }

//       if (rejectionReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Rejection message cannot contain more than 1000 characters",
//         });
//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointmentsCollection = database.collection("appointments");
//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       if (role === "doctor") {
//         const currentUserId = getDoctorDocumentId(currentUser);
//         if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
//           res.status(403).json({
//             success: false,
//             message: "You can update only your own appointments",
//           });
//           return;
//         }
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be changed",
//         });
//         return;
//       }

//       if (requestedStatus === "completed" && currentStatus !== "approved") {
//         res.status(409).json({
//           success: false,
//           message: "Only an approved appointment can be marked as completed",
//         });
//         return;
//       }

//       if (
//         requestedStatus === "approved" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or rejected appointment can be approved",
//         });
//         return;
//       }

//       if (
//         requestedStatus === "rejected" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "approved"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or approved appointment can be rejected",
//         });
//         return;
//       }

//       if (requestedStatus === "approved" && currentStatus === "rejected") {
//         const anotherActiveAppointment = await appointmentsCollection.findOne({
//           _id: {
//             $ne: appointment._id,
//           },
//           doctorId: getDoctorString(appointment.doctorId),
//           patientUserId: getDoctorString(appointment.patientUserId),
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "This patient already has another pending or approved appointment with you.",
//           });
//           return;
//         }
//       }

//       const now = new Date();

//       const statusFields: Record<string, unknown> = {
//         status: requestedStatus as AppointmentStatus,
//         rejectionReason:
//           requestedStatus === "rejected" ? rejectionReason : null,
//         updatedAt: now,
//       };

//       if (requestedStatus === "approved") {
//         statusFields.approvedAt = now;
//         statusFields.rejectedAt = null;
//         statusFields.rejectionReason = null;
//       }

//       if (requestedStatus === "completed") {
//         statusFields.completedAt = now;
//       }

//       if (requestedStatus === "rejected") {
//         statusFields.rejectedAt = now;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: statusFields,
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message:
//           requestedStatus === "approved"
//             ? "Appointment approved successfully"
//             : requestedStatus === "completed"
//               ? "Consultation completed successfully."
//               : "Appointment rejected successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update appointment status error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment status",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin Dashboard Statistics                               <-- ADD THIS HERE
// ========================================================= */

// app.get(
//   "/api/v1/admin/dashboard/stats",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     console.log("✅ Admin dashboard stats route called!"); // Debug log
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       // Get total patients
//       const totalPatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient" });

//       // Get active patients
//       const activePatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient", status: "active" });

//       // Get blocked patients
//       const blockedPatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient", status: "blocked" });

//       // Get total doctors
//       const totalDoctors = await database
//         .collection("doctors")
//         .countDocuments();

//       // Get active doctors
//       const activeDoctors = await database
//         .collection("doctors")
//         .countDocuments({ status: "active" });

//       // Get blocked doctors
//       const blockedDoctors = await database
//         .collection("doctors")
//         .countDocuments({ status: "blocked" });

//       // Get appointment counts by status
//       const appointmentCounts = await database
//         .collection("appointments")
//         .aggregate([
//           {
//             $group: {
//               _id: "$status",
//               count: { $sum: 1 },
//             },
//           },
//         ])
//         .toArray();

//       // Create status count object with default values
//       const statusCounts: Record<string, number> = {
//         pending: 0,
//         approved: 0,
//         completed: 0,
//         rejected: 0,
//       };

//       appointmentCounts.forEach((item) => {
//         const status = item._id || "pending";
//         if (status in statusCounts) {
//           statusCounts[status] = item.count;
//         }
//       });

//       // Get total appointments
//       const totalAppointments = await database
//         .collection("appointments")
//         .countDocuments();

//       // Get completed consultations
//       const completedConsultations = statusCounts.completed;

//       // Get monthly appointment trends (last 6 months)
//       const sixMonthsAgo = new Date();
//       sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

//       const monthlyTrends = await database
//         .collection("appointments")
//         .aggregate([
//           {
//             $match: {
//               createdAt: { $gte: sixMonthsAgo },
//             },
//           },
//           {
//             $group: {
//               _id: {
//                 year: { $year: "$createdAt" },
//                 month: { $month: "$createdAt" },
//               },
//               count: { $sum: 1 },
//               pending: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
//                 },
//               },
//               approved: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
//                 },
//               },
//               completed: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
//                 },
//               },
//               rejected: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
//                 },
//               },
//             },
//           },
//           {
//             $sort: { "_id.year": 1, "_id.month": 1 },
//           },
//         ])
//         .toArray();

//       // Get appointment status breakdown for charts
//       const statusColors: Record<string, string> = {
//         pending: "#FBBF24",
//         approved: "#60A5FA",
//         completed: "#34D399",
//         rejected: "#F87171",
//       };

//       const statusData = appointmentCounts.map((item) => ({
//         name: item._id || "unknown",
//         value: item.count,
//         fill: statusColors[item._id] || "#9CA3AF",
//       }));

//       // Format monthly data for charts
//       const monthNames = [
//         "Jan",
//         "Feb",
//         "Mar",
//         "Apr",
//         "May",
//         "Jun",
//         "Jul",
//         "Aug",
//         "Sep",
//         "Oct",
//         "Nov",
//         "Dec",
//       ];
//       const monthlyData = monthlyTrends.map((item) => ({
//         month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
//         pending: item.pending || 0,
//         approved: item.approved || 0,
//         completed: item.completed || 0,
//         rejected: item.rejected || 0,
//         total: item.count || 0,
//       }));

//       // Get recent appointments (last 10)
//       const recentAppointments = await database
//         .collection("appointments")
//         .find()
//         .sort({ createdAt: -1 })
//         .limit(10)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         data: {
//           overview: {
//             totalPatients,
//             activePatients,
//             blockedPatients,
//             totalDoctors,
//             activeDoctors,
//             blockedDoctors,
//             totalAppointments,
//             completedConsultations,
//             appointmentStatus: statusCounts,
//           },
//           charts: {
//             appointmentStatus: statusData,
//             monthlyTrends: monthlyData,
//           },
//           recentAppointments: recentAppointments.map((app) => ({
//             id: app._id,
//             patientName: app.patientName,
//             doctorName: app.doctorName,
//             specialization: app.specialization,
//             appointmentDate: app.appointmentDate,
//             appointmentTime: app.appointmentTime,
//             status: app.status,
//             createdAt: app.createdAt,
//           })),
//         },
//       });
//     } catch (error) {
//       console.error("Dashboard stats error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to fetch dashboard statistics",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin Dashboard Statistics                         <-- ADD THIS HERE
// ========================================================= */

// // app.get(
// //   "/api/v1/admin/dashboard/stats",
// //   verifyToken,
// //   verifyAdmin,
// //   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
// //     console.log("✅ Admin dashboard stats route called!"); // Debug log
// //     try {
// //       if (!database) {
// //         res.status(503).json({
// //           success: false,
// //           message: "Database is not connected",
// //         });
// //         return;
// //       }

// //       // Get total patients
// //       const totalPatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient" });

// //       // Get active patients
// //       const activePatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient", status: "active" });

// //       // Get blocked patients
// //       const blockedPatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient", status: "blocked" });

// //       // Get total doctors
// //       const totalDoctors = await database
// //         .collection("doctors")
// //         .countDocuments();

// //       // Get active doctors
// //       const activeDoctors = await database
// //         .collection("doctors")
// //         .countDocuments({ status: "active" });

// //       // Get blocked doctors
// //       const blockedDoctors = await database
// //         .collection("doctors")
// //         .countDocuments({ status: "blocked" });

// //       // Get appointment counts by status
// //       const appointmentCounts = await database
// //         .collection("appointments")
// //         .aggregate([
// //           {
// //             $group: {
// //               _id: "$status",
// //               count: { $sum: 1 },
// //             },
// //           },
// //         ])
// //         .toArray();

// //       // Create status count object with default values
// //       const statusCounts: Record<string, number> = {
// //         pending: 0,
// //         approved: 0,
// //         completed: 0,
// //         rejected: 0,
// //       };

// //       appointmentCounts.forEach((item) => {
// //         const status = item._id || "pending";
// //         if (status in statusCounts) {
// //           statusCounts[status] = item.count;
// //         }
// //       });

// //       // Get total appointments
// //       const totalAppointments = await database
// //         .collection("appointments")
// //         .countDocuments();

// //       // Get completed consultations
// //       const completedConsultations = statusCounts.completed;

// //       // Get monthly appointment trends (last 6 months)
// //       const sixMonthsAgo = new Date();
// //       sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

// //       const monthlyTrends = await database
// //         .collection("appointments")
// //         .aggregate([
// //           {
// //             $match: {
// //               createdAt: { $gte: sixMonthsAgo },
// //             },
// //           },
// //           {
// //             $group: {
// //               _id: {
// //                 year: { $year: "$createdAt" },
// //                 month: { $month: "$createdAt" },
// //               },
// //               count: { $sum: 1 },
// //               pending: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
// //                 },
// //               },
// //               approved: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
// //                 },
// //               },
// //               completed: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
// //                 },
// //               },
// //               rejected: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
// //                 },
// //               },
// //             },
// //           },
// //           {
// //             $sort: { "_id.year": 1, "_id.month": 1 },
// //           },
// //         ])
// //         .toArray();

// //       // Get appointment status breakdown for charts
// //       const statusColors: Record<string, string> = {
// //         pending: "#FBBF24",
// //         approved: "#60A5FA",
// //         completed: "#34D399",
// //         rejected: "#F87171",
// //       };

// //       const statusData = appointmentCounts.map((item) => ({
// //         name: item._id || "unknown",
// //         value: item.count,
// //         fill: statusColors[item._id] || "#9CA3AF",
// //       }));

// //       // Format monthly data for charts
// //       const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// //       const monthlyData = monthlyTrends.map((item) => ({
// //         month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
// //         pending: item.pending || 0,
// //         approved: item.approved || 0,
// //         completed: item.completed || 0,
// //         rejected: item.rejected || 0,
// //         total: item.count || 0,
// //       }));

// //       // Get recent appointments (last 10)
// //       const recentAppointments = await database
// //         .collection("appointments")
// //         .find()
// //         .sort({ createdAt: -1 })
// //         .limit(10)
// //         .toArray();

// //       res.status(200).json({
// //         success: true,
// //         data: {
// //           overview: {
// //             totalPatients,
// //             activePatients,
// //             blockedPatients,
// //             totalDoctors,
// //             activeDoctors,
// //             blockedDoctors,
// //             totalAppointments,
// //             completedConsultations,
// //             appointmentStatus: statusCounts,
// //           },
// //           charts: {
// //             appointmentStatus: statusData,
// //             monthlyTrends: monthlyData,
// //           },
// //           recentAppointments: recentAppointments.map((app) => ({
// //             id: app._id,
// //             patientName: app.patientName,
// //             doctorName: app.doctorName,
// //             specialization: app.specialization,
// //             appointmentDate: app.appointmentDate,
// //             appointmentTime: app.appointmentTime,
// //             status: app.status,
// //             createdAt: app.createdAt,
// //           })),
// //         },
// //       });
// //     } catch (error) {
// //       console.error("Dashboard stats error:", error);
// //       res.status(500).json({
// //         success: false,
// //         message: "Failed to fetch dashboard statistics",
// //       });
// //     }
// //   },
// // );

// /* =========================================================
//    SebaSathi AI Health Assistant (Groq)
// ========================================================= */

// type AIHealthMessageRole = "user" | "assistant";

// type AIHealthUrgency = "routine" | "soon" | "urgent" | "emergency";

// type AIHealthStreamStage =
//   | "thinking"
//   | "tool"
//   | "answering"
//   | "structuring"
//   | "saving";

// interface AIHealthMessage {
//   role: AIHealthMessageRole;
//   content: string;
// }

// interface AIHealthNavigationRoute {
//   label: string;
//   href: string;
//   description: string;
// }

// interface AIHealthNavigationAction {
//   label: string;
//   href: string;
//   reason: string;
// }

// interface AIHealthAssistantResponse {
//   reply: string;
//   urgencyLevel: AIHealthUrgency;
//   suggestedSpecialists: string[];
//   recommendedActions: string[];
//   warningSigns: string[];
//   followUpQuestions: string[];
//   suggestedPrompts: string[];
//   navigationActions: AIHealthNavigationAction[];
//   decisionBasis: string;
//   toolsUsed: string[];
//   contextMemoryUsed: boolean;
//   disclaimer: string;
// }

// interface AIHealthStoredMessage extends AIHealthMessage {
//   id: string;
//   assistant?: AIHealthAssistantResponse;
//   createdAt: Date;
// }

// interface AIHealthSummaryReport {
//   reportTitle: string;
//   conciseSummary: string;
//   chiefConcerns: string[];
//   symptoms: string[];
//   durationAndPattern: string;
//   severity: string;
//   urgencyLevel: AIHealthUrgency;
//   redFlags: string[];
//   suggestedSpecialists: string[];
//   selfCareGuidance: string[];
//   questionsForDoctor: string[];
//   emergencyAdvice: string;
//   disclaimer: string;
// }

// interface AIHealthConversationDocument {
//   _id?: ObjectId;
//   title?: string;
//   messages: AIHealthStoredMessage[];
//   summaryHistoryId?: string | null;
//   summaryReport?: AIHealthSummaryReport | null;
//   updatedAt?: Date;
//   lastMessageAt?: Date;
// }

// interface AIHealthApplicationContext {
//   user: {
//     id: string;
//     name: string;
//     role: UserRole;
//   };
//   routes: AIHealthNavigationRoute[];
//   doctorDirectory: {
//     activeDoctorCount: number;
//     specializations: string[];
//     highlightedDoctors: Array<{
//       id: string;
//       name: string;
//       specialization: string;
//       hospital: string;
//       ratingAverage: number;
//     }>;
//   } | null;
//   appointmentContext: {
//     total: number;
//     counts: Record<string, number>;
//     recentAppointments: Array<{
//       id: string;
//       doctorName: string;
//       patientName: string;
//       specialization: string;
//       appointmentDate: string;
//       appointmentTime: string;
//       status: string;
//     }>;
//   } | null;
//   recentHealthHistory: Array<{
//     id: string;
//     title: string;
//     urgencyLevel: AIHealthUrgency;
//     updatedAt: string | null;
//   }>;
//   toolsUsed: string[];
//   contextMemoryUsed: boolean;
// }

// const aiHealthRateLimit = new Map<
//   string,
//   {
//     startedAt: number;
//     count: number;
//   }
// >();

// const verifyAIHealthRateLimit = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): void => {
//   const key = req.userId || req.ip || "anonymous";
//   const now = Date.now();
//   const windowLength = 10 * 60 * 1000;
//   const maximumRequests = 40;
//   const current = aiHealthRateLimit.get(key);

//   if (!current || now - current.startedAt >= windowLength) {
//     aiHealthRateLimit.set(key, {
//       startedAt: now,
//       count: 1,
//     });

//     next();
//     return;
//   }

//   if (current.count >= maximumRequests) {
//     res.status(429).json({
//       success: false,
//       message:
//         "You have sent too many AI requests. Please try again after a few minutes.",
//       code: "AI_RATE_LIMITED",
//     });

//     return;
//   }

//   current.count += 1;
//   aiHealthRateLimit.set(key, current);
//   next();
// };

// const createAIHealthMessageId = (): string => {
//   return new ObjectId().toHexString();
// };

// const getAIHealthArray = (value: unknown, maximumItems = 8): string[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   return value
//     .map((item) => getDoctorString(item))
//     .filter(Boolean)
//     .slice(0, maximumItems);
// };

// const getAIHealthUrgency = (value: unknown): AIHealthUrgency => {
//   return value === "soon" || value === "urgent" || value === "emergency"
//     ? value
//     : "routine";
// };

// const getAIHealthBoolean = (value: unknown): boolean => value === true;

// const extractAIHealthJson = (content: string): Record<string, unknown> => {
//   const trimmed = content.trim();
//   const withoutFence = trimmed
//     .replace(/^```(?:json)?\s*/i, "")
//     .replace(/\s*```$/i, "")
//     .trim();

//   try {
//     const parsed = JSON.parse(withoutFence) as unknown;

//     if (typeof parsed === "object" && parsed !== null) {
//       return parsed as Record<string, unknown>;
//     }
//   } catch {
//     const firstBrace = withoutFence.indexOf("{");
//     const lastBrace = withoutFence.lastIndexOf("}");

//     if (firstBrace >= 0 && lastBrace > firstBrace) {
//       const parsed = JSON.parse(
//         withoutFence.slice(firstBrace, lastBrace + 1),
//       ) as unknown;

//       if (typeof parsed === "object" && parsed !== null) {
//         return parsed as Record<string, unknown>;
//       }
//     }
//   }

//   throw new Error("Groq returned an invalid structured response");
// };

// const normalizeAIHealthMessages = (
//   value: unknown,
//   options: {
//     requireLatestUser: boolean;
//     maximumMessages?: number;
//     maximumCharacters?: number;
//   },
// ):
//   | {
//       success: true;
//       messages: AIHealthMessage[];
//     }
//   | {
//       success: false;
//       message: string;
//     } => {
//   if (!Array.isArray(value)) {
//     return {
//       success: false,
//       message: "A conversation message list is required",
//     };
//   }

//   const maximumMessages = options.maximumMessages ?? 30;
//   const maximumCharacters = options.maximumCharacters ?? 30000;
//   const messages: AIHealthMessage[] = [];
//   let totalCharacters = 0;

//   for (const rawMessage of value.slice(-maximumMessages)) {
//     if (typeof rawMessage !== "object" || rawMessage === null) {
//       continue;
//     }

//     const message = rawMessage as Record<string, unknown>;
//     const role = message.role;
//     const content = getDoctorString(message.content);

//     if ((role !== "user" && role !== "assistant") || !content) {
//       continue;
//     }

//     if (content.length > 4000) {
//       return {
//         success: false,
//         message: "Each chat message cannot contain more than 4000 characters",
//       };
//     }

//     totalCharacters += content.length;

//     if (totalCharacters > maximumCharacters) {
//       return {
//         success: false,
//         message:
//           "This conversation is too long. Please generate a summary and start a new conversation.",
//       };
//     }

//     messages.push({
//       role,
//       content,
//     });
//   }

//   if (messages.length === 0) {
//     return {
//       success: false,
//       message: "At least one valid chat message is required",
//     };
//   }

//   if (!messages.some((message) => message.role === "user")) {
//     return {
//       success: false,
//       message: "At least one user message is required",
//     };
//   }

//   if (
//     options.requireLatestUser &&
//     messages[messages.length - 1]?.role !== "user"
//   ) {
//     return {
//       success: false,
//       message: "The latest conversation message must be from the user",
//     };
//   }

//   return {
//     success: true,
//     messages,
//   };
// };

// const PUBLIC_AI_HEALTH_NAVIGATION_ROUTES: AIHealthNavigationRoute[] = [
//   {
//     label: "Home",
//     href: "/",
//     description: "Open the SebaSathi home page.",
//   },
//   {
//     label: "Find Doctors",
//     href: "/find-doctors",
//     description: "Find active doctors and filter by specialization.",
//   },
//   {
//     label: "AI Health Assistant",
//     href: "/ai-health-assistant",
//     description: "Continue using the SebaSathi AI Health Assistant.",
//   },
//   {
//     label: "About Us",
//     href: "/about",
//     description: "Learn more about SebaSathi and its healthcare services.",
//   },
//   {
//     label: "Contact",
//     href: "/contact",
//     description: "Open the SebaSathi contact page.",
//   },
// ];

// const ROLE_AI_HEALTH_NAVIGATION_ROUTES: Record<
//   UserRole,
//   AIHealthNavigationRoute[]
// > = {
//   patient: [
//     {
//       label: "Patient Overview",
//       href: "/dashboard/patient",
//       description: "Open the patient's dashboard overview.",
//     },
//     {
//       label: "My Appointments",
//       href: "/dashboard/patient/appointments",
//       description: "View the patient's appointment requests and statuses.",
//     },
//     {
//       label: "Prescriptions",
//       href: "/dashboard/patient/prescriptions",
//       description: "View the patient's saved prescriptions.",
//     },
//     {
//       label: "Consultations",
//       href: "/dashboard/patient/consultations",
//       description: "View the patient's consultation records.",
//     },
//     {
//       label: "AI Health History",
//       href: "/dashboard/patient/ai-health-history",
//       description: "Review saved AI-generated health summaries.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/patient/my-profile",
//       description: "Open the patient's profile settings.",
//     },
//   ],
//   doctor: [
//     {
//       label: "Doctor Overview",
//       href: "/dashboard/doctor",
//       description: "Open the doctor's dashboard overview.",
//     },
//     {
//       label: "Appointments",
//       href: "/dashboard/doctor/patients-appointments",
//       description: "View appointments assigned to the signed-in doctor.",
//     },
//     {
//       label: "My Patients",
//       href: "/dashboard/doctor/patients",
//       description: "View the doctor's patient list.",
//     },
//     {
//       label: "Prescriptions",
//       href: "/dashboard/doctor/prescriptions",
//       description: "Create or review doctor prescription records.",
//     },
//     {
//       label: "Consultation Records",
//       href: "/dashboard/doctor/consultations",
//       description: "View the doctor's consultation records.",
//     },
//     {
//       label: "Availability",
//       href: "/dashboard/doctor/availability",
//       description: "Manage the doctor's availability schedule.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/doctor/my-profile",
//       description: "Open the doctor's profile settings.",
//     },
//   ],
//   admin: [
//     {
//       label: "Admin Overview",
//       href: "/dashboard/admin",
//       description: "Open the administrator dashboard overview.",
//     },
//     {
//       label: "Manage Users",
//       href: "/dashboard/admin/users",
//       description: "Open administrator user management.",
//     },
//     {
//       label: "Manage Doctors",
//       href: "/dashboard/admin/doctors",
//       description: "Open administrator doctor management.",
//     },
//     {
//       label: "Manage Appointments",
//       href: "/dashboard/admin/appointments",
//       description: "Open administrator appointment management.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/admin/my-profile",
//       description: "Open the administrator's profile settings.",
//     },
//   ],
// };

// const AI_HEALTH_NAVIGATION_ROUTE_ALIASES: Record<string, string> = {
//   "/doctors": "/find-doctors",
//   "/dashboard/doctor/appointments": "/dashboard/doctor/patients-appointments",
// };

// const normalizeAIHealthNavigationHref = (href: string): string => {
//   return AI_HEALTH_NAVIGATION_ROUTE_ALIASES[href] || href;
// };

// const getAllAIHealthNavigationRoutes = (): AIHealthNavigationRoute[] => [
//   ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.patient,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.doctor,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.admin,
// ];

// const getAIHealthNavigationRoutes = (
//   role: UserRole,
// ): AIHealthNavigationRoute[] => [
//   ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES[role],
// ];

// const getAIHealthNavigationActions = (
//   value: unknown,
//   allowedRoutes: AIHealthNavigationRoute[],
// ): AIHealthNavigationAction[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   const allowedByHref = new Map(
//     allowedRoutes.map((route) => [route.href, route] as const),
//   );

//   const actions: AIHealthNavigationAction[] = [];

//   for (const rawAction of value) {
//     if (typeof rawAction !== "object" || rawAction === null) {
//       continue;
//     }

//     const action = rawAction as Record<string, unknown>;
//     const href = normalizeAIHealthNavigationHref(getDoctorString(action.href));
//     const allowedRoute = allowedByHref.get(href);

//     if (!allowedRoute) {
//       continue;
//     }

//     actions.push({
//       label: getDoctorString(action.label) || allowedRoute.label,
//       href,
//       reason: getDoctorString(action.reason) || allowedRoute.description,
//     });

//     if (actions.length >= 3) {
//       break;
//     }
//   }

//   return actions;
// };

// const formatAIHealthAssistantResponse = (
//   data: Record<string, unknown>,
//   emergencyDetected: boolean,
//   context?: AIHealthApplicationContext,
// ): AIHealthAssistantResponse => {
//   const urgencyLevel = emergencyDetected
//     ? "emergency"
//     : getAIHealthUrgency(data.urgencyLevel);

//   const reply =
//     getDoctorString(data.reply) ||
//     "Please describe the symptoms, duration and severity a little more clearly.";

//   const followUpQuestions = getAIHealthArray(data.followUpQuestions, 3);
//   const suggestedPrompts = getAIHealthArray(data.suggestedPrompts, 4);
//   const allowedRoutes = context?.routes || getAllAIHealthNavigationRoutes();
//   const toolsUsed = context?.toolsUsed.length
//     ? context.toolsUsed
//     : getAIHealthArray(data.toolsUsed, 8);
//   const contextMemoryUsed = context
//     ? context.contextMemoryUsed
//     : getAIHealthBoolean(data.contextMemoryUsed);

//   return {
//     reply,
//     urgencyLevel,
//     suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 3),
//     recommendedActions: getAIHealthArray(data.recommendedActions, 5),
//     warningSigns: emergencyDetected
//       ? Array.from(
//           new Set([
//             "Your description may include an emergency warning sign.",
//             ...getAIHealthArray(data.warningSigns, 4),
//           ]),
//         ).slice(0, 4)
//       : getAIHealthArray(data.warningSigns, 4),
//     followUpQuestions,
//     suggestedPrompts:
//       suggestedPrompts.length > 0
//         ? suggestedPrompts
//         : followUpQuestions.slice(0, 3),
//     navigationActions: getAIHealthNavigationActions(
//       data.navigationActions,
//       allowedRoutes,
//     ),
//     decisionBasis:
//       getDoctorString(data.decisionBasis) ||
//       "This guidance is based on the symptoms, duration, severity, warning signs and relevant SebaSathi application context available in this conversation.",
//     toolsUsed,
//     contextMemoryUsed,
//     disclaimer:
//       getDoctorString(data.disclaimer) ||
//       "General guidance only; this is not a diagnosis or prescription.",
//   };
// };

// const getStoredAIHealthMessages = (value: unknown): AIHealthStoredMessage[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   return value
//     .map((rawMessage): AIHealthStoredMessage | null => {
//       if (typeof rawMessage !== "object" || rawMessage === null) {
//         return null;
//       }

//       const message = rawMessage as Record<string, unknown>;
//       const role = message.role;
//       const content = getDoctorString(message.content);

//       if ((role !== "user" && role !== "assistant") || !content) {
//         return null;
//       }

//       const assistant =
//         typeof message.assistant === "object" && message.assistant !== null
//           ? formatAIHealthAssistantResponse(
//               message.assistant as Record<string, unknown>,
//               false,
//             )
//           : undefined;

//       const createdAtValue = message.createdAt;
//       const createdAt =
//         createdAtValue instanceof Date
//           ? createdAtValue
//           : new Date(
//               typeof createdAtValue === "string" ||
//                 typeof createdAtValue === "number"
//                 ? createdAtValue
//                 : Date.now(),
//             );

//       return {
//         id: getDoctorString(message.id) || createAIHealthMessageId(),
//         role,
//         content,
//         ...(assistant ? { assistant } : {}),
//         createdAt: Number.isNaN(createdAt.getTime()) ? new Date() : createdAt,
//       };
//     })
//     .filter((message): message is AIHealthStoredMessage => message !== null);
// };

// const hasEmergencyWarning = (messages: AIHealthMessage[]): boolean => {
//   const text = messages
//     .filter((message) => message.role === "user")
//     .map((message) => message.content)
//     .join(" ")
//     .toLowerCase();

//   const emergencyPatterns = [
//     /severe chest pain/,
//     /cannot breathe/,
//     /can't breathe/,
//     /difficulty breathing/,
//     /heavy bleeding/,
//     /unconscious/,
//     /not responding/,
//     /seizure/,
//     /stroke symptoms/,
//     /face droop/,
//     /suicid(?:e|al)/,
//     /kill myself/,
//     /বুকে তীব্র ব্যথা/,
//     /শ্বাস নিতে পারছি না/,
//     /শ্বাসকষ্ট/,
//     /অতিরিক্ত রক্তপাত/,
//     /অজ্ঞান/,
//     /খিঁচুনি/,
//     /আত্মহত্যা/,
//   ];

//   return emergencyPatterns.some((pattern) => pattern.test(text));
// };

// const callGroqAI = async (
//   messages: Array<{
//     role: "system" | "user" | "assistant";
//     content: string;
//   }>,
//   temperature: number,
//   maximumOutputTokens: number,
// ): Promise<Record<string, unknown>> => {
//   if (!groqApiKey) {
//     throw new Error("GROQ_API_KEY is missing from the backend .env file");
//   }

//   const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
//     method: "POST",
//     headers: {
//       authorization: `Bearer ${groqApiKey}`,
//       "content-type": "application/json",
//       accept: "application/json",
//     },
//     body: JSON.stringify({
//       model: groqModel,
//       messages,
//       temperature,
//       max_completion_tokens: maximumOutputTokens,
//       response_format: {
//         type: "json_object",
//       },
//     }),
//   });

//   const responseData = (await response.json().catch(() => null)) as Record<
//     string,
//     unknown
//   > | null;

//   if (!response.ok) {
//     const errorObject =
//       typeof responseData?.error === "object" && responseData.error !== null
//         ? (responseData.error as Record<string, unknown>)
//         : null;

//     const providerMessage = getDoctorString(errorObject?.message);

//     throw new Error(
//       providerMessage || `Groq request failed with status ${response.status}`,
//     );
//   }

//   const choices = Array.isArray(responseData?.choices)
//     ? responseData.choices
//     : [];

//   const firstChoice = choices[0];

//   if (typeof firstChoice !== "object" || firstChoice === null) {
//     throw new Error("Groq did not return an assistant response");
//   }

//   const choice = firstChoice as Record<string, unknown>;
//   const message =
//     typeof choice.message === "object" && choice.message !== null
//       ? (choice.message as Record<string, unknown>)
//       : null;

//   const content = getDoctorString(message?.content);

//   if (!content) {
//     throw new Error("Groq returned an empty assistant response");
//   }

//   return extractAIHealthJson(content);
// };

// const callGroqTextStream = async (
//   messages: Array<{
//     role: "system" | "user" | "assistant";
//     content: string;
//   }>,
//   onDelta: (delta: string) => void,
//   signal?: AbortSignal,
// ): Promise<string> => {
//   if (!groqApiKey) {
//     throw new Error("GROQ_API_KEY is missing from the backend .env file");
//   }

//   const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
//     method: "POST",
//     headers: {
//       authorization: `Bearer ${groqApiKey}`,
//       "content-type": "application/json",
//       accept: "text/event-stream",
//     },
//     body: JSON.stringify({
//       model: groqModel,
//       messages,
//       temperature: 0.25,
//       max_completion_tokens: 1100,
//       stream: true,
//     }),
//     signal,
//   });

//   if (!response.ok) {
//     const responseData = (await response.json().catch(() => null)) as Record<
//       string,
//       unknown
//     > | null;
//     const errorObject =
//       typeof responseData?.error === "object" && responseData.error !== null
//         ? (responseData.error as Record<string, unknown>)
//         : null;
//     const providerMessage = getDoctorString(errorObject?.message);

//     throw new Error(
//       providerMessage || `Groq request failed with status ${response.status}`,
//     );
//   }

//   if (!response.body) {
//     throw new Error("Groq streaming response body is unavailable");
//   }

//   const reader = response.body.getReader();
//   const decoder = new TextDecoder();
//   let buffer = "";
//   let completeText = "";

//   while (true) {
//     const { value, done } = await reader.read();

//     if (done) {
//       break;
//     }

//     buffer += decoder.decode(value, { stream: true });
//     const lines = buffer.split("\n");
//     buffer = lines.pop() || "";

//     for (const line of lines) {
//       const trimmed = line.trim();

//       if (!trimmed.startsWith("data:")) {
//         continue;
//       }

//       const payload = trimmed.slice(5).trim();

//       if (!payload || payload === "[DONE]") {
//         continue;
//       }

//       const parsed = JSON.parse(payload) as Record<string, unknown>;
//       const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
//       const firstChoice = choices[0];

//       if (typeof firstChoice !== "object" || firstChoice === null) {
//         continue;
//       }

//       const delta = (firstChoice as Record<string, unknown>).delta;

//       if (typeof delta !== "object" || delta === null) {
//         continue;
//       }

//       const rawContent = (delta as Record<string, unknown>).content;
//       const content = typeof rawContent === "string" ? rawContent : "";

//       if (!content) {
//         continue;
//       }

//       completeText += content;
//       onDelta(content);
//     }
//   }

//   const finalText = completeText.trim();

//   if (!finalText) {
//     throw new Error("Groq returned an empty streamed response");
//   }

//   return finalText;
// };

// const formatAIHealthSummary = (
//   data: Record<string, unknown>,
// ): AIHealthSummaryReport => {
//   return {
//     reportTitle:
//       getDoctorString(data.reportTitle) || "AI Health Conversation Summary",
//     conciseSummary:
//       getDoctorString(data.conciseSummary) ||
//       "A concise summary could not be generated.",
//     chiefConcerns: getAIHealthArray(data.chiefConcerns, 6),
//     symptoms: getAIHealthArray(data.symptoms, 10),
//     durationAndPattern:
//       getDoctorString(data.durationAndPattern) || "Not clearly stated",
//     severity: getDoctorString(data.severity) || "Not clearly stated",
//     urgencyLevel: getAIHealthUrgency(data.urgencyLevel),
//     redFlags: getAIHealthArray(data.redFlags, 6),
//     suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 5),
//     selfCareGuidance: getAIHealthArray(data.selfCareGuidance, 6),
//     questionsForDoctor: getAIHealthArray(data.questionsForDoctor, 6),
//     emergencyAdvice:
//       getDoctorString(data.emergencyAdvice) ||
//       "Seek urgent in-person medical care if symptoms become severe or new warning signs appear.",
//     disclaimer:
//       getDoctorString(data.disclaimer) ||
//       "This AI-generated summary is not a diagnosis or prescription.",
//   };
// };

// const createAIHealthConversationTitle = (message: string): string => {
//   const normalized = message.replace(/\s+/g, " ").trim();
//   const words = normalized.split(" ").filter(Boolean).slice(0, 7);
//   const title = words.join(" ");

//   if (!title) {
//     return "New health chat";
//   }

//   return normalized.length > title.length ? `${title}…` : title;
// };

// const getAIHealthOwnerFilter = (userId: string): Filter<Document> => {
//   return {
//     $or: [
//       {
//         userId,
//       },
//       {
//         patientUserId: userId,
//       },
//     ],
//   };
// };

// const formatAIHealthConversationMessage = (message: AIHealthStoredMessage) => {
//   return {
//     id: message.id,
//     role: message.role,
//     content: message.content,
//     assistant: message.assistant || null,
//     createdAt: formatDoctorDate(message.createdAt),
//   };
// };

// const formatAIHealthConversation = (conversation: Document) => {
//   const userId =
//     getDoctorString(conversation.userId) ||
//     getDoctorString(conversation.patientUserId);

//   const userRole: UserRole =
//     conversation.userRole === "admin" ||
//     conversation.userRole === "doctor" ||
//     conversation.userRole === "patient"
//       ? conversation.userRole
//       : "patient";

//   const messages = getStoredAIHealthMessages(conversation.messages);

//   return {
//     id: getDoctorDocumentId(conversation),
//     title: getDoctorString(conversation.title) || "New health chat",
//     userId,
//     userRole,
//     userName:
//       getDoctorString(conversation.userName) ||
//       getDoctorString(conversation.patientName),
//     userEmail:
//       normalizeDoctorEmail(conversation.userEmail) ||
//       normalizeDoctorEmail(conversation.patientEmail),
//     userImage:
//       getDoctorString(conversation.userImage) ||
//       getDoctorString(conversation.patientImage) ||
//       null,
//     messages: messages.map(formatAIHealthConversationMessage),
//     messageCount: messages.length,
//     summaryHistoryId: getDoctorString(conversation.summaryHistoryId) || null,
//     summaryReport:
//       typeof conversation.summaryReport === "object" &&
//       conversation.summaryReport !== null
//         ? conversation.summaryReport
//         : null,
//     createdAt: formatDoctorDate(conversation.createdAt),
//     updatedAt: formatDoctorDate(conversation.updatedAt),
//     lastMessageAt: formatDoctorDate(
//       conversation.lastMessageAt || conversation.updatedAt,
//     ),
//   };
// };

// const formatAIHealthHistory = (history: Document) => {
//   const userId =
//     getDoctorString(history.userId) || getDoctorString(history.patientUserId);

//   const userName =
//     getDoctorString(history.userName) || getDoctorString(history.patientName);

//   const userEmail =
//     normalizeDoctorEmail(history.userEmail) ||
//     normalizeDoctorEmail(history.patientEmail);

//   const userRole: UserRole =
//     history.userRole === "admin" ||
//     history.userRole === "doctor" ||
//     history.userRole === "patient"
//       ? history.userRole
//       : "patient";

//   return {
//     id: getDoctorDocumentId(history),
//     conversationId: getDoctorString(history.conversationId) || null,
//     conversationTitle: getDoctorString(history.conversationTitle) || null,
//     userId,
//     userRole,
//     userName,
//     userEmail,
//     userImage:
//       getDoctorString(history.userImage) ||
//       getDoctorString(history.patientImage) ||
//       null,
//     patientUserId: userId,
//     patientName: userName,
//     patientEmail: userEmail,
//     provider: getDoctorString(history.provider),
//     model: getDoctorString(history.model),
//     report:
//       typeof history.report === "object" && history.report !== null
//         ? history.report
//         : null,
//     messages: Array.isArray(history.messages) ? history.messages : [],
//     createdAt: formatDoctorDate(history.createdAt),
//     updatedAt: formatDoctorDate(history.updatedAt),
//   };
// };

// const getAIHealthConversationForUser = async (
//   userId: string,
//   conversationId: string,
// ): Promise<Document | null> => {
//   if (!database || !conversationId) {
//     return null;
//   }

//   return database.collection(AI_HEALTH_CHAT_COLLECTION).findOne({
//     $and: [getDoctorFilter(conversationId), getAIHealthOwnerFilter(userId)],
//   });
// };

// const detectAIHealthApplicationIntents = (message: string) => {
//   const normalized = message.toLowerCase();

//   return {
//     appointment:
//       /appointment|booking|schedule|pending|approved|rejected|অ্যাপয়েন্টমেন্ট|অ্যাপয়েন্টমেন্ট|বুকিং|সিডিউল|পেন্ডিং|এপ্রুভ/.test(
//         normalized,
//       ),
//     history:
//       /history|summary|report|previous chat|old chat|হিস্ট্রি|সামারি|রিপোর্ট|পুরোনো চ্যাট/.test(
//         normalized,
//       ),
//     navigation:
//       /open|go to|take me|navigate|where is|show page|dashboard|খুলে দাও|নিয়ে যাও|নিয়ে যাও|কোথায়|কোথায়|ড্যাশবোর্ড/.test(
//         normalized,
//       ),
//     doctor:
//       /doctor|specialist|specialization|cardio|derma|neuro|medicine|surgeon|ডাক্তার|বিশেষজ্ঞ|স্পেশালিস্ট|কার্ডিও|ডার্মা|নিউরো/.test(
//         normalized,
//       ),
//   };
// };

// const buildAIHealthApplicationContext = async (
//   currentUser: Document,
//   latestMessage: string,
//   existingMessages: AIHealthStoredMessage[],
// ): Promise<AIHealthApplicationContext> => {
//   if (!database) {
//     throw new Error("Database is not connected");
//   }

//   const userId = getDoctorDocumentId(currentUser);
//   const role = getNormalizedUserRole(currentUser);
//   const userName = getDoctorString(currentUser.name) || "User";
//   const intents = detectAIHealthApplicationIntents(latestMessage);
//   const routes = getAIHealthNavigationRoutes(role);
//   const toolsUsed = ["SebaSathi navigation map", "SebaSathi doctor directory"];
//   const contextMemoryUsed = existingMessages.length > 0;

//   if (contextMemoryUsed) {
//     toolsUsed.push("Conversation memory");
//   }

//   const [doctorDocuments, specializations, activeDoctorCount] =
//     await Promise.all([
//       database
//         .collection("doctors")
//         .find(
//           { status: "active" },
//           {
//             projection: {
//               name: 1,
//               specialization: 1,
//               hospital: 1,
//               chamber: 1,
//               ratingAverage: 1,
//             },
//           },
//         )
//         .sort({ ratingAverage: -1, ratingCount: -1, createdAt: -1 })
//         .limit(8)
//         .toArray(),
//       database.collection("doctors").distinct("specialization", {
//         status: "active",
//       }),
//       database.collection("doctors").countDocuments({ status: "active" }),
//     ]);

//   const doctorDirectory = {
//     activeDoctorCount,
//     specializations: specializations
//       .map((value) => getDoctorString(value))
//       .filter(Boolean)
//       .sort((a, b) => a.localeCompare(b))
//       .slice(0, 30),
//     highlightedDoctors: doctorDocuments.map((doctor) => ({
//       id: getDoctorDocumentId(doctor),
//       name: getDoctorString(doctor.name),
//       specialization: getDoctorString(doctor.specialization),
//       hospital:
//         getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
//       ratingAverage: Number.isFinite(Number(doctor.ratingAverage))
//         ? Number(Number(doctor.ratingAverage).toFixed(1))
//         : 0,
//     })),
//   };

//   let appointmentContext: AIHealthApplicationContext["appointmentContext"] =
//     null;

//   if (intents.appointment || intents.navigation) {
//     toolsUsed.push("Appointment lookup");

//     const appointmentFilter: Filter<Document> =
//       role === "patient"
//         ? { patientUserId: userId }
//         : role === "doctor"
//           ? { doctorUserId: userId }
//           : {};

//     const [statusCounts, recentAppointments, total] = await Promise.all([
//       database
//         .collection("appointments")
//         .aggregate([
//           { $match: appointmentFilter },
//           {
//             $group: {
//               _id: "$status",
//               count: { $sum: 1 },
//             },
//           },
//         ])
//         .toArray(),
//       database
//         .collection("appointments")
//         .find(appointmentFilter)
//         .sort({ updatedAt: -1, createdAt: -1 })
//         .limit(5)
//         .toArray(),
//       database.collection("appointments").countDocuments(appointmentFilter),
//     ]);

//     appointmentContext = {
//       total,
//       counts: Object.fromEntries(
//         statusCounts.map((item) => [
//           getDoctorString(item._id) || "unknown",
//           Number(item.count) || 0,
//         ]),
//       ),
//       recentAppointments: recentAppointments.map((appointment) => ({
//         id: getDoctorDocumentId(appointment),
//         doctorName: getDoctorString(appointment.doctorName),
//         patientName: getDoctorString(appointment.patientName),
//         specialization: getDoctorString(appointment.specialization),
//         appointmentDate: getDoctorString(appointment.appointmentDate),
//         appointmentTime: getDoctorString(appointment.appointmentTime),
//         status: getDoctorString(appointment.status) || "pending",
//       })),
//     };
//   }

//   let recentHealthHistory: AIHealthApplicationContext["recentHealthHistory"] =
//     [];

//   if (intents.history || intents.navigation) {
//     toolsUsed.push("Saved AI health history lookup");

//     const historyDocuments = await database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .find(getAIHealthOwnerFilter(userId))
//       .sort({ updatedAt: -1, createdAt: -1 })
//       .limit(5)
//       .toArray();

//     recentHealthHistory = historyDocuments.map((history) => {
//       const report =
//         typeof history.report === "object" && history.report !== null
//           ? (history.report as Record<string, unknown>)
//           : {};

//       return {
//         id: getDoctorDocumentId(history),
//         title:
//           getDoctorString(history.conversationTitle) ||
//           getDoctorString(report.reportTitle) ||
//           "AI Health Summary",
//         urgencyLevel: getAIHealthUrgency(report.urgencyLevel),
//         updatedAt: formatDoctorDate(history.updatedAt || history.createdAt),
//       };
//     });
//   }

//   if (intents.doctor) {
//     toolsUsed.push("Specialist matching context");
//   }

//   return {
//     user: {
//       id: userId,
//       name: userName,
//       role,
//     },
//     routes,
//     doctorDirectory,
//     appointmentContext,
//     recentHealthHistory,
//     toolsUsed: Array.from(new Set(toolsUsed)),
//     contextMemoryUsed,
//   };
// };

// const buildAIHealthNaturalResponsePrompt = (
//   context: AIHealthApplicationContext,
// ): string => `You are SebaSathi AI Health Assistant, an advanced conversational assistant integrated into a Bangladesh-oriented healthcare application.

// You must do more than simple text generation. Use conversation memory and the supplied SebaSathi application context to answer questions, reason about next steps, help the user navigate the application, and ask useful follow-up questions when information is missing.

// Signed-in user:
// ${JSON.stringify(context.user)}

// SebaSathi application context retrieved by backend tools:
// ${JSON.stringify({
//   routes: context.routes,
//   doctorDirectory: context.doctorDirectory,
//   appointmentContext: context.appointmentContext,
//   recentHealthHistory: context.recentHealthHistory,
//   toolsUsed: context.toolsUsed,
// })}

// Behavior requirements:
// - Answer health questions and SebaSathi application questions naturally.
// - Use previous conversation messages to understand references such as “it”, “that problem”, “same pain”, or “what should I do next”.
// - When application data is available, use it accurately. Never invent appointments, doctors, history, counts, status, dates or routes.
// - If the user asks where to go in the application, explain the correct page and mention the relevant route label naturally.
// - Explain the practical basis for recommendations without revealing hidden chain-of-thought.
// - Ask concise follow-up questions when key details are missing.
// - Match the user's language: easy Bangla, Banglish or English.
// - For health guidance, never confirm a diagnosis, prescribe medicine, provide individualized doses, or advise stopping prescribed treatment.
// - Emergency warning signs require immediate emergency-care advice.
// - Usually write 5-9 clear sentences and approximately 120-240 words when enough information exists.
// - Return only the natural conversational answer. Do not return JSON, markdown tables, internal IDs or hidden reasoning.`;

// const buildAIHealthMetadataPrompt = (
//   context: AIHealthApplicationContext,
//   latestUserMessage: string,
//   assistantReply: string,
// ): string => `Create safe structured metadata for a completed SebaSathi AI assistant reply.

// User message:
// ${latestUserMessage}

// Assistant reply:
// ${assistantReply}

// Allowed navigation routes:
// ${JSON.stringify(context.routes)}

// Backend tools already used:
// ${JSON.stringify(context.toolsUsed)}

// Return ONLY valid JSON with this exact shape:
// {
//   "urgencyLevel": "routine | soon | urgent | emergency",
//   "suggestedSpecialists": ["maximum three specialist categories that exist in or reasonably map to the doctor directory"],
//   "recommendedActions": ["maximum five safe practical actions"],
//   "warningSigns": ["maximum four important warning signs"],
//   "followUpQuestions": ["maximum three useful follow-up questions"],
//   "suggestedPrompts": ["maximum four short prompts the user can click to continue the conversation"],
//   "navigationActions": [
//     {
//       "label": "must correspond to an allowed route",
//       "href": "must exactly match one allowed route href",
//       "reason": "short explanation of why this page is relevant"
//     }
//   ],
//   "decisionBasis": "one or two concise user-facing sentences explaining which reported facts or application context influenced the guidance, without exposing private chain-of-thought",
//   "toolsUsed": ["copy only tools actually listed above"],
//   "contextMemoryUsed": ${context.contextMemoryUsed ? "true" : "false"},
//   "disclaimer": "one short medical disclaimer"
// }

// Do not invent app data. Include navigationActions only when useful. Suggested prompts must be directly usable as the user's next message.`;

// const writeAIHealthStreamEvent = (
//   res: Response,
//   event: Record<string, unknown>,
// ): void => {
//   if (!res.writableEnded) {
//     res.write(`${JSON.stringify(event)}\n`);
//   }
// };

// const startAIHealthStream = (res: Response): void => {
//   res.status(200);
//   res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
//   res.setHeader("Cache-Control", "no-cache, no-transform");
//   res.setHeader("Connection", "keep-alive");
//   res.setHeader("X-Accel-Buffering", "no");
//   res.flushHeaders();
// };

// const writeAIHealthStatus = (
//   res: Response,
//   stage: AIHealthStreamStage,
//   message: string,
//   toolsUsed: string[] = [],
// ): void => {
//   writeAIHealthStreamEvent(res, {
//     type: "status",
//     stage,
//     message,
//     toolsUsed,
//   });
// };

// /* =========================================================
//    AI Health access status
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/access",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           allowed: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);
//       const allowed = status === "active";

//       res.status(200).json({
//         success: true,
//         authenticated: true,
//         allowed,
//         role,
//         status,
//         user: {
//           id: getDoctorDocumentId(currentUser),
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//         },
//         message: allowed
//           ? "Your active account can use SebaSathi AI Health Assistant."
//           : "Your account is blocked. Contact the administrator to use the AI Health Assistant.",
//       });
//     } catch (error) {
//       console.error("AI Health access error:", error);

//       res.status(500).json({
//         success: false,
//         allowed: false,
//         message: "Failed to verify AI Health access",
//       });
//     }
//   },
// );

// /* =========================================================
//    AI Health conversation history
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/conversations",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const limit = getPositiveInteger(req.query.limit, 100, 100);
//       const conversations = await database
//         .collection(AI_HEALTH_CHAT_COLLECTION)
//         .find(getAIHealthOwnerFilter(userId))
//         .sort({
//           lastMessageAt: -1,
//           updatedAt: -1,
//           _id: -1,
//         })
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         conversations: conversations.map(formatAIHealthConversation),
//       });
//     } catch (error) {
//       console.error("Get AI conversations error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI conversations",
//       });
//     }
//   },
// );

// app.post(
//   "/api/v1/ai-health/conversations",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const now = new Date();
//       const userId = getDoctorDocumentId(currentUser);
//       const userRole = getNormalizedUserRole(currentUser);
//       const userName = getDoctorString(currentUser.name);
//       const userEmail = normalizeDoctorEmail(currentUser.email);
//       const userImage = getDoctorString(currentUser.image) || null;
//       const requestedTitle = getDoctorString(req.body.title).slice(0, 80);

//       const conversationDocument = {
//         title: requestedTitle || "New health chat",
//         userId,
//         userRole,
//         userName,
//         userEmail,
//         userImage,
//         patientUserId: userId,
//         patientName: userName,
//         patientEmail: userEmail,
//         patientImage: userImage,
//         messages: [] as AIHealthStoredMessage[],
//         summaryHistoryId: null,
//         summaryReport: null,
//         createdAt: now,
//         updatedAt: now,
//         lastMessageAt: now,
//       };

//       const collection = database.collection(AI_HEALTH_CHAT_COLLECTION);
//       const insertResult = await collection.insertOne(conversationDocument);
//       const conversation = await collection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "New AI health conversation created",
//         conversation: conversation
//           ? formatAIHealthConversation(conversation)
//           : {
//               id: insertResult.insertedId.toHexString(),
//               ...conversationDocument,
//               messageCount: 0,
//             },
//       });
//     } catch (error) {
//       console.error("Create AI conversation error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to create AI conversation",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/ai-health/conversations/:conversationId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         getDoctorDocumentId(currentUser),
//         getDoctorString(req.params.conversationId),
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         conversation: formatAIHealthConversation(conversation),
//       });
//     } catch (error) {
//       console.error("Get AI conversation details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI conversation",
//       });
//     }
//   },
// );

// app.delete(
//   "/api/v1/ai-health/conversations/:conversationId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       await database.collection(AI_HEALTH_CHAT_COLLECTION).deleteOne({
//         _id: conversation._id,
//       });

//       res.status(200).json({
//         success: true,
//         message: "AI health conversation deleted successfully",
//         deletedConversationId: conversationId,
//       });
//     } catch (error) {
//       console.error("Delete AI conversation error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to delete AI conversation",
//       });
//     }
//   },
// );

// /* =========================================================
//    Advanced streamed AI Health message exchange
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/conversations/:conversationId/messages/stream",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     let streamStarted = false;
//     const abortController = new AbortController();

//     res.on("close", () => {
//       if (!res.writableEnded) {
//         abortController.abort();
//       }
//     });

//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const content = getDoctorString(req.body.message);

//       if (!content) {
//         res.status(400).json({
//           success: false,
//           message: "A health or application question is required",
//         });
//         return;
//       }

//       if (content.length > 4000) {
//         res.status(400).json({
//           success: false,
//           message: "A message cannot contain more than 4000 characters",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       const existingMessages = getStoredAIHealthMessages(conversation.messages);
//       const contextMessages: AIHealthMessage[] = [
//         ...existingMessages.map(({ role, content: savedContent }) => ({
//           role,
//           content: savedContent,
//         })),
//         {
//           role: "user",
//           content,
//         },
//       ];

//       const validation = normalizeAIHealthMessages(contextMessages, {
//         requireLatestUser: true,
//         maximumMessages: 26,
//         maximumCharacters: 32000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       startAIHealthStream(res);
//       streamStarted = true;
//       writeAIHealthStatus(
//         res,
//         "thinking",
//         "Understanding your question and previous conversation...",
//       );

//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         content,
//         existingMessages,
//       );

//       writeAIHealthStatus(
//         res,
//         "tool",
//         "Checking relevant SebaSathi context...",
//         applicationContext.toolsUsed,
//       );

//       writeAIHealthStatus(
//         res,
//         "answering",
//         "Preparing a context-aware response...",
//         applicationContext.toolsUsed,
//       );

//       const naturalReply = await callGroqTextStream(
//         [
//           {
//             role: "system",
//             content: buildAIHealthNaturalResponsePrompt(applicationContext),
//           },
//           ...validation.messages,
//         ],
//         (delta) => {
//           writeAIHealthStreamEvent(res, {
//             type: "delta",
//             delta,
//           });
//         },
//         abortController.signal,
//       );

//       writeAIHealthStatus(
//         res,
//         "structuring",
//         "Creating follow-up prompts, navigation actions and decision support...",
//         applicationContext.toolsUsed,
//       );

//       let metadata: Record<string, unknown> = {};

//       try {
//         metadata = await callGroqAI(
//           [
//             {
//               role: "system",
//               content: buildAIHealthMetadataPrompt(
//                 applicationContext,
//                 content,
//                 naturalReply,
//               ),
//             },
//             {
//               role: "user",
//               content: "Return the requested JSON metadata now.",
//             },
//           ],
//           0.1,
//           900,
//         );
//       } catch (metadataError) {
//         console.error(
//           "AI Health metadata generation warning:",
//           metadataError instanceof Error
//             ? metadataError.message
//             : metadataError,
//         );
//       }

//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const assistant = formatAIHealthAssistantResponse(
//         {
//           ...metadata,
//           reply: naturalReply,
//           toolsUsed: applicationContext.toolsUsed,
//           contextMemoryUsed: applicationContext.contextMemoryUsed,
//         },
//         emergencyDetected,
//         applicationContext,
//       );

//       const now = new Date();
//       const userMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "user",
//         content,
//         createdAt: now,
//       };
//       const assistantMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "assistant",
//         content: naturalReply,
//         assistant,
//         createdAt: new Date(),
//       };

//       const nextTitle =
//         existingMessages.some((message) => message.role === "user") ||
//         getDoctorString(conversation.title) !== "New health chat"
//           ? getDoctorString(conversation.title) || "New health chat"
//           : createAIHealthConversationTitle(content);

//       writeAIHealthStatus(
//         res,
//         "saving",
//         "Saving the conversation and memory...",
//         applicationContext.toolsUsed,
//       );

//       const updatedConversation = await database
//         .collection<AIHealthConversationDocument>(AI_HEALTH_CHAT_COLLECTION)
//         .findOneAndUpdate(
//           {
//             _id: conversation._id,
//           },
//           {
//             $set: {
//               title: nextTitle,
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: assistantMessage.createdAt,
//               lastMessageAt: assistantMessage.createdAt,
//             },
//             $push: {
//               messages: {
//                 $each: [userMessage, assistantMessage],
//               },
//             },
//           },
//           {
//             returnDocument: "after",
//           },
//         );

//       writeAIHealthStreamEvent(res, {
//         type: "result",
//         data: {
//           success: true,
//           provider: "groq",
//           model: groqModel,
//           userMessage: formatAIHealthConversationMessage(userMessage),
//           assistantMessage: formatAIHealthConversationMessage(assistantMessage),
//           conversation: updatedConversation
//             ? formatAIHealthConversation(updatedConversation)
//             : null,
//         },
//       });

//       res.end();
//     } catch (error) {
//       console.error("AI Health streamed chat error:", error);

//       const message =
//         error instanceof Error
//           ? error.message
//           : "Failed to receive a streamed response from the AI provider";

//       if (streamStarted) {
//         writeAIHealthStreamEvent(res, {
//           type: "error",
//           message,
//         });
//         res.end();
//       } else {
//         res.status(502).json({
//           success: false,
//           message,
//         });
//       }
//     }
//   },
// );

// /* =========================================================
//    Non-streaming persistent message exchange compatibility
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/conversations/:conversationId/messages",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const content = getDoctorString(req.body.message);

//       if (!content) {
//         res.status(400).json({
//           success: false,
//           message: "A health or application question is required",
//         });
//         return;
//       }

//       if (content.length > 4000) {
//         res.status(400).json({
//           success: false,
//           message: "A message cannot contain more than 4000 characters",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       const existingMessages = getStoredAIHealthMessages(conversation.messages);
//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         content,
//         existingMessages,
//       );
//       const contextMessages: AIHealthMessage[] = [
//         ...existingMessages.map(({ role, content: savedContent }) => ({
//           role,
//           content: savedContent,
//         })),
//         {
//           role: "user",
//           content,
//         },
//       ];
//       const validation = normalizeAIHealthMessages(contextMessages, {
//         requireLatestUser: true,
//         maximumMessages: 26,
//         maximumCharacters: 32000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const groqData = await callGroqAI(
//         [
//           {
//             role: "system",
//             content: `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn ONLY JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`,
//           },
//           ...validation.messages,
//         ],
//         0.2,
//         1200,
//       );

//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const assistant = formatAIHealthAssistantResponse(
//         groqData,
//         emergencyDetected,
//         applicationContext,
//       );
//       const now = new Date();
//       const userMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "user",
//         content,
//         createdAt: now,
//       };
//       const assistantMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "assistant",
//         content: assistant.reply,
//         assistant,
//         createdAt: new Date(),
//       };
//       const nextTitle =
//         existingMessages.some((message) => message.role === "user") ||
//         getDoctorString(conversation.title) !== "New health chat"
//           ? getDoctorString(conversation.title) || "New health chat"
//           : createAIHealthConversationTitle(content);

//       const updatedConversation = await database
//         .collection<AIHealthConversationDocument>(AI_HEALTH_CHAT_COLLECTION)
//         .findOneAndUpdate(
//           { _id: conversation._id },
//           {
//             $set: {
//               title: nextTitle,
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: assistantMessage.createdAt,
//               lastMessageAt: assistantMessage.createdAt,
//             },
//             $push: {
//               messages: {
//                 $each: [userMessage, assistantMessage],
//               },
//             },
//           },
//           { returnDocument: "after" },
//         );

//       res.status(200).json({
//         success: true,
//         provider: "groq",
//         model: groqModel,
//         userMessage: formatAIHealthConversationMessage(userMessage),
//         assistantMessage: formatAIHealthConversationMessage(assistantMessage),
//         conversation: updatedConversation
//           ? formatAIHealthConversation(updatedConversation)
//           : null,
//       });
//     } catch (error) {
//       console.error("AI Health persistent chat error:", error);

//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to receive a response from the AI provider",
//       });
//     }
//   },
// );

// /* =========================================================
//    Legacy AI Health chat endpoint
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/chat",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const validation = normalizeAIHealthMessages(req.body.messages, {
//         requireLatestUser: true,
//         maximumMessages: 22,
//         maximumCharacters: 26000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const latestMessage = validation.messages.at(-1)?.content || "";
//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         latestMessage,
//         [],
//       );
//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const systemPrompt = `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn only JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`;

//       const groqData = await callGroqAI(
//         [{ role: "system", content: systemPrompt }, ...validation.messages],
//         0.2,
//         1200,
//       );

//       res.status(200).json({
//         success: true,
//         provider: "groq",
//         model: groqModel,
//         assistant: formatAIHealthAssistantResponse(
//           groqData,
//           emergencyDetected,
//           applicationContext,
//         ),
//       });
//     } catch (error) {
//       console.error("AI Health chat error:", error);
//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to receive a response from the AI provider",
//       });
//     }
//   },
// );

// /* =========================================================
//    Generate and save AI Health summary
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/summary",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.body.conversationId);
//       let conversation: Document | null = null;
//       let conversationTitle = "AI Health Conversation";
//       let messagesValue: unknown = req.body.messages;

//       if (conversationId) {
//         conversation = await getAIHealthConversationForUser(
//           userId,
//           conversationId,
//         );

//         if (!conversation) {
//           res.status(404).json({
//             success: false,
//             message: "AI health conversation was not found",
//           });
//           return;
//         }

//         conversationTitle =
//           getDoctorString(conversation.title) || "AI Health Conversation";
//         messagesValue = getStoredAIHealthMessages(conversation.messages).map(
//           ({ role, content }) => ({ role, content }),
//         );
//       }

//       const validation = normalizeAIHealthMessages(messagesValue, {
//         requireLatestUser: false,
//         maximumMessages: 40,
//         maximumCharacters: 42000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const systemPrompt = `Generate a concise structured health-conversation report for SebaSathi AI. Use only information actually present. Do not invent symptoms, duration, tests, diagnoses or medicines. Do not diagnose or prescribe. Match the user's language where practical.

// Return ONLY valid JSON:
// {
//   "reportTitle": "short title",
//   "conciseSummary": "2-3 concise sentences",
//   "chiefConcerns": ["main concerns"],
//   "symptoms": ["reported symptoms"],
//   "durationAndPattern": "stated duration/pattern or Not clearly stated",
//   "severity": "stated severity or Not clearly stated",
//   "urgencyLevel": "routine | soon | urgent | emergency",
//   "redFlags": ["warning signs"],
//   "suggestedSpecialists": ["specialist categories"],
//   "selfCareGuidance": ["low-risk general guidance"],
//   "questionsForDoctor": ["useful questions"],
//   "emergencyAdvice": "brief emergency advice",
//   "disclaimer": "not a diagnosis or prescription"
// }`;

//       const groqData = await callGroqAI(
//         [
//           {
//             role: "system",
//             content: systemPrompt,
//           },
//           {
//             role: "user",
//             content: JSON.stringify(validation.messages),
//           },
//         ],
//         0.1,
//         1100,
//       );

//       const report = formatAIHealthSummary(groqData);
//       const now = new Date();
//       const userRole = getNormalizedUserRole(currentUser);
//       const userName = getDoctorString(currentUser.name);
//       const userEmail = normalizeDoctorEmail(currentUser.email);
//       const userImage = getDoctorString(currentUser.image) || null;
//       const historyCollection = database.collection(
//         AI_HEALTH_HISTORY_COLLECTION,
//       );

//       const historyDocument = {
//         conversationId: conversation ? getDoctorDocumentId(conversation) : null,
//         conversationTitle,
//         userId,
//         userRole,
//         userName,
//         userEmail,
//         userImage,
//         patientUserId: userId,
//         patientName: userName,
//         patientEmail: userEmail,
//         patientImage: userImage,
//         provider: "groq",
//         model: groqModel,
//         report,
//         messages: validation.messages,
//         createdAt: now,
//         updatedAt: now,
//       };

//       let history: Document | null = null;
//       const existingSummaryId = getDoctorString(conversation?.summaryHistoryId);

//       if (existingSummaryId) {
//         const existingHistory = await historyCollection.findOne({
//           $and: [
//             getDoctorFilter(existingSummaryId),
//             getAIHealthOwnerFilter(userId),
//           ],
//         });

//         if (existingHistory) {
//           history = await historyCollection.findOneAndUpdate(
//             { _id: existingHistory._id },
//             {
//               $set: {
//                 ...historyDocument,
//                 createdAt: existingHistory.createdAt || now,
//                 updatedAt: now,
//               },
//             },
//             { returnDocument: "after" },
//           );
//         }
//       }

//       if (!history) {
//         const insertResult = await historyCollection.insertOne(historyDocument);
//         history = await historyCollection.findOne({
//           _id: insertResult.insertedId,
//         });
//       }

//       if (!history) {
//         throw new Error("The generated summary could not be saved");
//       }

//       if (conversation) {
//         await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
//           { _id: conversation._id },
//           {
//             $set: {
//               summaryHistoryId: getDoctorDocumentId(history),
//               summaryReport: report,
//               updatedAt: now,
//             },
//           },
//         );
//       }

//       res.status(201).json({
//         success: true,
//         message: "AI health summary generated and saved successfully",
//         history: formatAIHealthHistory(history),
//         conversation: conversation
//           ? formatAIHealthConversation({
//               ...conversation,
//               summaryHistoryId: getDoctorDocumentId(history),
//               summaryReport: report,
//               updatedAt: now,
//             })
//           : null,
//       });
//     } catch (error) {
//       console.error("AI Health summary error:", error);

//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to generate and save the AI health summary",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient AI Health summary history
//    - Active and blocked patients can read their own history.
//    - Only active patients can delete their own history.
// ========================================================= */

// app.get(
//   "/api/v1/patient/ai-health-history",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const status = getNormalizedUserStatus(currentUser);
//       const requestedPage = getPositiveInteger(req.query.page, 1, 100000);

//       // Patient AI Health History always returns exactly 10 records per page.
//       const limit = 10;
//       const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
//       const filter = getAIHealthOwnerFilter(patientUserId);
//       const total = await collection.countDocuments(filter);
//       const totalPages = Math.max(1, Math.ceil(total / limit));
//       const page = Math.min(requestedPage, totalPages);

//       const documents = await collection
//         .find(filter)
//         .sort({
//           updatedAt: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .skip((page - 1) * limit)
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         account: {
//           id: patientUserId,
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//           role: "patient",
//           status,
//         },
//         canDelete: status === "active",
//         histories: documents.map(formatAIHealthHistory),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages,
//         },
//       });
//     } catch (error) {
//       console.error("Get patient AI Health history error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient AI health history",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/patient/ai-health-history/:historyId",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);

//       if (!historyId) {
//         res.status(400).json({
//           success: false,
//           message: "AI health history ID is required",
//         });
//         return;
//       }

//       const history = await database
//         .collection(AI_HEALTH_HISTORY_COLLECTION)
//         .findOne({
//           $and: [
//             getDoctorFilter(historyId),
//             getAIHealthOwnerFilter(patientUserId),
//           ],
//         });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       const status = getNormalizedUserStatus(currentUser);

//       res.status(200).json({
//         success: true,
//         account: {
//           id: patientUserId,
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//           role: "patient",
//           status,
//         },
//         canDelete: status === "active",
//         history: formatAIHealthHistory(history),
//       });
//     } catch (error) {
//       console.error("Get patient AI Health history details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient AI health history details",
//       });
//     }
//   },
// );

// app.delete(
//   "/api/v1/patient/ai-health-history/:historyId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);

//       if (!historyId) {
//         res.status(400).json({
//           success: false,
//           message: "AI health history ID is required",
//         });
//         return;
//       }

//       const historyCollection = database.collection(
//         AI_HEALTH_HISTORY_COLLECTION,
//       );

//       const history = await historyCollection.findOne({
//         $and: [
//           getDoctorFilter(historyId),
//           getAIHealthOwnerFilter(patientUserId),
//         ],
//       });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       const deleteResult = await historyCollection.deleteOne({
//         _id: history._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "AI health history could not be deleted",
//         });
//         return;
//       }

//       const conversationId = getDoctorString(history.conversationId);

//       if (conversationId) {
//         await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
//           {
//             $and: [
//               getDoctorFilter(conversationId),
//               getAIHealthOwnerFilter(patientUserId),
//               {
//                 summaryHistoryId: historyId,
//               },
//             ],
//           },
//           {
//             $set: {
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: new Date(),
//             },
//           },
//         );
//       }

//       res.status(200).json({
//         success: true,
//         message: "AI health history deleted successfully",
//         deletedHistoryId: historyId,
//       });
//     } catch (error) {
//       console.error("Delete patient AI Health history error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete patient AI health history",
//       });
//     }
//   },
// );

// /* =========================================================
//    Saved AI Health summary history
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/history",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;
//       const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
//       const filter = getAIHealthOwnerFilter(userId);

//       const [documents, total] = await Promise.all([
//         collection
//           .find(filter)
//           .sort({ createdAt: -1, _id: -1 })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),
//         collection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         histories: documents.map(formatAIHealthHistory),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get AI Health history error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI health history",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/ai-health/history/:historyId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);
//       const history = await database
//         .collection(AI_HEALTH_HISTORY_COLLECTION)
//         .findOne({
//           $and: [getDoctorFilter(historyId), getAIHealthOwnerFilter(userId)],
//         });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         history: formatAIHealthHistory(history),
//       });
//     } catch (error) {
//       console.error("Get AI Health history details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI health history details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Unknown route handler
// ========================================================= */

// app.use((_req: Request, res: Response) => {
//   res.status(404).json({
//     success: false,
//     message: "API route not found",
//   });
// });

// /* =========================================================
//    Global error handler
// ========================================================= */

// app.use((error: Error, _req: Request, res: Response, _next: NextFunction) => {
//   console.error("Server error:", error);

//   res.status(500).json({
//     success: false,
//     message: "Internal server error",
//   });
// });

// /* =========================================================
//    MongoDB connection
// ========================================================= */

// const connectDatabase = async (): Promise<void> => {
//   await mongoClient.connect();

//   database = mongoClient.db(mongoDbName);

//   await database.command({ ping: 1 });

//   await Promise.all([
//     database
//       .collection("user")
//       .createIndex({ role: 1, status: 1, updatedAt: -1 }),
//     database.collection("user").createIndex({ role: 1, name: 1 }),
//     database.collection("user").createIndex({ role: 1, email: 1 }),
//     database
//       .collection("doctors")
//       .createIndex({ status: 1, ratingAverage: -1, createdAt: -1 }),
//     database.collection("doctors").createIndex({ name: 1 }),
//     database.collection("doctors").createIndex({ specialization: 1 }),
//     database.collection("doctors").createIndex({ qualification: 1 }),
//     database.collection("doctors").createIndex({ hospital: 1 }),
//     database.collection("doctors").createIndex({ experienceYears: 1 }),
//     database
//       .collection("reviews")
//       .createIndex({ doctorId: 1, userId: 1 }, { unique: true }),
//     database.collection("reviews").createIndex({ doctorId: 1, updatedAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ patientUserId: 1, doctorId: 1, status: 1 }),
//     database
//       .collection("appointments")
//       .createIndex({ patientUserId: 1, createdAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ doctorUserId: 1, status: 1, appointmentDate: 1 }),
//     database
//       .collection("appointments")
//       .createIndex({ doctorUserId: 1, createdAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ status: 1, appointmentDate: 1, appointmentTime: 1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ userId: 1, createdAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ patientUserId: 1, createdAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ userId: 1, updatedAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ patientUserId: 1, updatedAt: -1 }),
//     database
//       .collection(AI_HEALTH_CHAT_COLLECTION)
//       .createIndex({ userId: 1, lastMessageAt: -1 }),
//     database
//       .collection(AI_HEALTH_CHAT_COLLECTION)
//       .createIndex({ patientUserId: 1, lastMessageAt: -1 }),
//   ]);

//   console.log(`MongoDB connected successfully. Database: ${mongoDbName}`);
// };

// /* =========================================================
//    Start server
// ========================================================= */

// const startServer = async (): Promise<void> => {
//   try {
//     await connectDatabase();

//     app.listen(port, () => {
//       console.log(`SebaSathi AI server is running on http://localhost:${port}`);

//       console.log(`JWKS URL: ${jwksUrl.toString()}`);
//     });
//   } catch (error) {
//     console.error(
//       "Unable to start SebaSathi AI server:",
//       error instanceof Error ? error.message : error,
//     );

//     await mongoClient.close();
//     process.exit(1);
//   }
// };

// void startServer();

// /* =========================================================
//    Graceful shutdown
// ========================================================= */

// const shutdownServer = async (signal: string): Promise<void> => {
//   console.log(`${signal} received. Closing MongoDB connection...`);

//   try {
//     await mongoClient.close();

//     console.log("MongoDB connection closed successfully");

//     process.exit(0);
//   } catch (error) {
//     console.error("Error closing MongoDB connection:", error);

//     process.exit(1);
//   }
// };

// process.on("SIGINT", () => {
//   void shutdownServer("SIGINT");
// });

// process.on("SIGTERM", () => {
//   void shutdownServer("SIGTERM");
// });

// export default app;

// import cors from "cors";
// import dotenv from "dotenv";
// import express, {
//   type NextFunction,
//   type Request,
//   type Response,
// } from "express";
// import { createRemoteJWKSet, jwtVerify } from "jose-cjs";
// import {
//   MongoClient,
//   ObjectId,
//   ServerApiVersion,
//   type Db,
//   type Document,
//   type Filter,
// } from "mongodb";

// dotenv.config({ quiet: true });

// /* =========================================================
//    Environment variables
// ========================================================= */

// const port = Number(process.env.PORT) || 5000;

// const clientUrl = process.env.CLIENT_URL || "http://localhost:3000";

// const betterAuthUrl = (
//   process.env.BETTER_AUTH_URL || "http://localhost:3000"
// ).replace(/\/+$/, "");

// const mongoDbUri = process.env.MONGODB_URI;
// const mongoDbName = process.env.MONGODB_DB_NAME;

// const groqApiKey = process.env.GROQ_API_KEY;

// const groqApiBaseUrl = (
//   process.env.GROQ_API_BASE_URL || "https://api.groq.com/openai/v1"
// ).replace(/\/+$/, "");

// const groqModel = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

// const AI_HEALTH_HISTORY_COLLECTION = "AI-health-History";

// const AI_HEALTH_CHAT_COLLECTION = "all-history";

// if (!mongoDbUri) {
//   throw new Error("MONGODB_URI is missing from the .env file");
// }

// if (!mongoDbName) {
//   throw new Error("MONGODB_DB_NAME is missing from the .env file");
// }

// /* =========================================================
//    Express application
// ========================================================= */

// const app = express();

// /* =========================================================
//    MongoDB configuration
// ========================================================= */

// const mongoClient = new MongoClient(mongoDbUri, {
//   serverApi: {
//     version: ServerApiVersion.v1,
//     strict: false,
//     deprecationErrors: true,
//   },
// });

// let database: Db | null = null;

// /* =========================================================
//    Better Auth JWKS configuration
// ========================================================= */

// const jwksUrl = new URL(`${betterAuthUrl}/api/auth/jwks`);

// const jwks = createRemoteJWKSet(jwksUrl);

// /* =========================================================
//    Authentication types
// ========================================================= */

// type UserRole = "admin" | "doctor" | "patient";
// type UserStatus = "active" | "blocked";

// interface AuthenticatedRequest extends Request {
//   userId?: string;
//   userName?: string;
//   userEmail?: string;
//   userRole?: UserRole;
//   userStatus?: UserStatus;
// }

// /* =========================================================
//    Global middlewares
// ========================================================= */

// app.use(
//   cors({
//     origin: clientUrl,
//     credentials: true,
//   }),
// );

// app.use(
//   express.json({
//     limit: "1mb",
//   }),
// );

// app.use(express.urlencoded({ extended: true }));

// /* =========================================================
//    JWT verification middleware
// ========================================================= */

// const verifyToken = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): Promise<void> => {
//   const authorizationHeader = req.headers.authorization;

//   if (!authorizationHeader) {
//     res.status(401).json({
//       success: false,
//       message: "Authorization token is required",
//     });

//     return;
//   }

//   const [authorizationType, token] = authorizationHeader.split(" ");

//   if (authorizationType !== "Bearer" || !token) {
//     res.status(401).json({
//       success: false,
//       message: "A valid Bearer token is required",
//     });

//     return;
//   }

//   try {
//     const { payload } = await jwtVerify(token, jwks);

//     const authenticatedUserId =
//       typeof payload.sub === "string"
//         ? payload.sub
//         : typeof payload.id === "string"
//           ? payload.id
//           : undefined;

//     if (!authenticatedUserId) {
//       res.status(403).json({
//         success: false,
//         message: "Token does not contain a valid user ID",
//       });

//       return;
//     }

//     req.userId = authenticatedUserId;

//     req.userName = typeof payload.name === "string" ? payload.name : undefined;

//     req.userEmail =
//       typeof payload.email === "string" ? payload.email : undefined;

//     next();
//   } catch (error) {
//     console.error(
//       "JWT verification error:",
//       error instanceof Error ? error.message : error,
//     );

//     res.status(403).json({
//       success: false,
//       message: "Invalid or expired access token",
//     });
//   }
// };

// /* =========================================================
//    Role verification middleware
// ========================================================= */

// const verifyRole = (requiredRole: UserRole) => {
//   return async (
//     req: AuthenticatedRequest,
//     res: Response,
//     next: NextFunction,
//   ): Promise<void> => {
//     try {
//       if (!req.userId) {
//         res.status(401).json({
//           success: false,
//           message: "Authentication is required before role verification",
//         });

//         return;
//       }

//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const userQueryConditions: Record<string, unknown>[] = [
//         {
//           id: req.userId,
//         },
//       ];

//       if (req.userEmail) {
//         userQueryConditions.push({
//           email: req.userEmail,
//         });
//       }

//       const currentUser = await usersCollection.findOne({
//         $or: userQueryConditions,
//       });

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       req.userStatus = currentUser.status === "blocked" ? "blocked" : "active";

//       const currentRole = currentUser.role;

//       const validRoles: UserRole[] = ["admin", "doctor", "patient"];

//       if (
//         typeof currentRole !== "string" ||
//         !validRoles.includes(currentRole as UserRole)
//       ) {
//         res.status(403).json({
//           success: false,
//           message: "User role is missing or invalid",
//         });

//         return;
//       }

//       if (currentRole !== requiredRole) {
//         res.status(403).json({
//           success: false,
//           message: `${requiredRole} access is required`,
//         });

//         return;
//       }

//       req.userRole = currentRole as UserRole;

//       next();
//     } catch (error) {
//       console.error(
//         "Role verification error:",
//         error instanceof Error ? error.message : error,
//       );

//       res.status(500).json({
//         success: false,
//         message: "Failed to verify current user role",
//       });
//     }
//   };
// };

// /* =========================================================
//    Admin, doctor and patient middlewares
// ========================================================= */

// const verifyAdmin = verifyRole("admin");

// const verifyDoctor = verifyRole("doctor");

// const verifyPatient = verifyRole("patient");

// /**
//  * Allows blocked users to read data, but prevents them
//  * from creating, editing, deleting, or changing status.
//  *
//  * Always use this after verifyRole().
//  */
// const verifyActive = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): void => {
//   if (req.userStatus !== "active") {
//     res.status(403).json({
//       success: false,
//       message:
//         "Your account is blocked. You can view data, but you cannot perform this action.",
//       code: "READ_ONLY_ACCOUNT",
//     });

//     return;
//   }

//   next();
// };

// /**
//  * Allows any authenticated role (admin, doctor or patient)
//  * to use protected features when the account status is active.
//  */
// const verifyAnyActiveUser = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): Promise<void> => {
//   try {
//     if (!req.userId) {
//       res.status(401).json({
//         success: false,
//         message: "Authentication is required",
//       });

//       return;
//     }

//     if (!database) {
//       res.status(503).json({
//         success: false,
//         message: "Database is not connected",
//       });

//       return;
//     }

//     const userQueryConditions: Record<string, unknown>[] = [
//       {
//         id: req.userId,
//       },
//     ];

//     if (req.userEmail) {
//       userQueryConditions.push({
//         email: req.userEmail.toLowerCase(),
//       });
//     }

//     const currentUser = await database.collection("user").findOne({
//       $or: userQueryConditions,
//     });

//     if (!currentUser) {
//       res.status(404).json({
//         success: false,
//         message: "User account was not found",
//       });

//       return;
//     }

//     const currentRole = currentUser.role;
//     const validRoles: UserRole[] = ["admin", "doctor", "patient"];

//     if (
//       typeof currentRole !== "string" ||
//       !validRoles.includes(currentRole as UserRole)
//     ) {
//       res.status(403).json({
//         success: false,
//         message: "User role is missing or invalid",
//       });

//       return;
//     }

//     const currentStatus: UserStatus =
//       currentUser.status === "blocked" ? "blocked" : "active";

//     req.userRole = currentRole as UserRole;
//     req.userStatus = currentStatus;

//     if (currentStatus !== "active") {
//       res.status(403).json({
//         success: false,
//         message:
//           "Your account is blocked. Only active accounts can use the AI Health Assistant.",
//         code: "READ_ONLY_ACCOUNT",
//       });

//       return;
//     }

//     next();
//   } catch (error) {
//     console.error(
//       "Active user verification error:",
//       error instanceof Error ? error.message : error,
//     );

//     res.status(500).json({
//       success: false,
//       message: "Failed to verify active user account",
//     });
//   }
// };

// /* =========================================================
//    Public root route
// ========================================================= */

// app.get("/", (_req: Request, res: Response) => {
//   res.status(200).json({
//     success: true,
//     message: "SebaSathi AI server is running",
//   });
// });

// /* =========================================================
//    Public health route
// ========================================================= */

// app.get("/api/v1/health", async (_req: Request, res: Response) => {
//   try {
//     if (!database) {
//       res.status(503).json({
//         success: false,
//         message: "Database is not connected",
//       });

//       return;
//     }

//     await database.command({ ping: 1 });

//     res.status(200).json({
//       success: true,
//       message: "SebaSathi AI API is healthy",
//       database: "connected",
//       databaseName: database.databaseName,
//       timestamp: new Date().toISOString(),
//     });
//   } catch {
//     res.status(503).json({
//       success: false,
//       message: "MongoDB connection is unavailable",
//       database: "disconnected",
//       timestamp: new Date().toISOString(),
//     });
//   }
// });

// /* =========================================================
//    Protected authentication test route
// ========================================================= */

// app.get(
//   "/api/v1/auth/me",
//   verifyToken,
//   (req: AuthenticatedRequest, res: Response) => {
//     res.status(200).json({
//       success: true,
//       message: "Authenticated user retrieved successfully",
//       user: {
//         id: req.userId,
//         name: req.userName || null,
//         email: req.userEmail || null,
//       },
//     });
//   },
// );

// /*
//   Admin API middleware:

//   app.get(
//     "/api/v1/admin/your-api",
//     verifyToken,
//     verifyAdmin,
//     yourController
//   );
// */

// /*
//   Doctor API middleware:

//   app.get(
//     "/api/v1/doctor/your-api",
//     verifyToken,
//     verifyDoctor,
//     yourController
//   );
// */

// /*
//   Patient API middleware:

//   app.get(
//     "/api/v1/patient/your-api",
//     verifyToken,
//     verifyPatient,
//     yourController
//   );
// */

// /* =========================================================
//    Current authenticated user
// ========================================================= */

// app.get(
//   "/api/users/current",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       if (!req.userId) {
//         res.status(401).json({
//           success: false,
//           message: "Authenticated user ID was not found",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const userQueryConditions: Record<string, unknown>[] = [
//         {
//           id: req.userId,
//         },
//       ];

//       if (req.userEmail) {
//         userQueryConditions.push({
//           email: req.userEmail.toLowerCase(),
//         });
//       }

//       const currentUser = await usersCollection.findOne({
//         $or: userQueryConditions,
//       });

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const currentRole: UserRole =
//         currentUser.role === "admin" ||
//         currentUser.role === "doctor" ||
//         currentUser.role === "patient"
//           ? currentUser.role
//           : "patient";

//       const currentStatus: "active" | "blocked" =
//         currentUser.status === "blocked" ? "blocked" : "active";

//       const currentUserId =
//         typeof currentUser.id === "string" && currentUser.id.trim()
//           ? currentUser.id
//           : currentUser._id instanceof ObjectId
//             ? currentUser._id.toHexString()
//             : req.userId;

//       res.status(200).json({
//         id: currentUserId,
//         _id: currentUserId,
//         name: typeof currentUser.name === "string" ? currentUser.name : null,
//         email: typeof currentUser.email === "string" ? currentUser.email : null,
//         image: typeof currentUser.image === "string" ? currentUser.image : null,
//         role: currentRole,
//         status: currentStatus,
//       });
//     } catch (error) {
//       console.error(
//         "Get current user error:",
//         error instanceof Error ? error.message : error,
//       );

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve current user",
//       });
//     }
//   },
// );

// /* =========================================================
//    Manage Doctors helpers
// ========================================================= */

// type DoctorStatus = "active" | "blocked";

// const getDoctorString = (value: unknown): string => {
//   return typeof value === "string" ? value.trim() : "";
// };

// const getDoctorNumber = (value: unknown): number => {
//   const numberValue =
//     typeof value === "number"
//       ? value
//       : typeof value === "string" && value.trim()
//         ? Number(value)
//         : 0;

//   return Number.isFinite(numberValue) ? Math.max(0, numberValue) : 0;
// };

// const normalizeDoctorEmail = (value: unknown): string => {
//   return getDoctorString(value).toLowerCase();
// };

// const isValidDoctorEmail = (email: string): boolean => {
//   return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
// };

// const escapeDoctorSearch = (value: string): string => {
//   return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
// };

// const getDoctorDocumentId = (document: Document): string => {
//   if (typeof document.id === "string" && document.id.trim()) {
//     return document.id;
//   }

//   if (document._id instanceof ObjectId) {
//     return document._id.toHexString();
//   }

//   return String(document._id || "");
// };

// const getDoctorFilter = (doctorId: string): Filter<Document> => {
//   const conditions: Filter<Document>[] = [
//     {
//       id: doctorId,
//     },
//   ];

//   if (ObjectId.isValid(doctorId)) {
//     conditions.push({
//       _id: new ObjectId(doctorId),
//     });
//   }

//   return {
//     $or: conditions,
//   };
// };

// const getUserFilter = (userId: string): Filter<Document> => {
//   const conditions: Filter<Document>[] = [
//     {
//       id: userId,
//     },
//   ];

//   if (ObjectId.isValid(userId)) {
//     conditions.push({
//       _id: new ObjectId(userId),
//     });
//   }

//   return {
//     $or: conditions,
//   };
// };

// const formatDoctorDate = (value: unknown): string | null => {
//   if (value instanceof Date) {
//     return value.toISOString();
//   }

//   if (typeof value === "string" || typeof value === "number") {
//     const date = new Date(value);

//     return Number.isNaN(date.getTime()) ? null : date.toISOString();
//   }

//   return null;
// };

// const formatDoctor = (doctor: Document) => {
//   return {
//     id: getDoctorDocumentId(doctor),

//     userId: typeof doctor.userId === "string" ? doctor.userId : "",

//     name: getDoctorString(doctor.name),

//     email: normalizeDoctorEmail(doctor.email),

//     image: getDoctorString(doctor.image) || null,

//     phone: getDoctorString(doctor.phone),

//     specialization: getDoctorString(doctor.specialization),

//     qualification: getDoctorString(doctor.qualification),

//     experienceYears: getDoctorNumber(doctor.experienceYears),

//     hospital:
//       getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),

//     address: getDoctorString(doctor.address),

//     bio: getDoctorString(doctor.bio),

//     role: "doctor" as const,

//     status:
//       doctor.status === "blocked" ? ("blocked" as const) : ("active" as const),

//     createdAt: formatDoctorDate(doctor.createdAt),

//     updatedAt: formatDoctorDate(doctor.updatedAt),
//   };
// };

// const readBetterAuthResponse = async (
//   response: globalThis.Response,
// ): Promise<unknown> => {
//   try {
//     return await response.json();
//   } catch {
//     return null;
//   }
// };

// const getBetterAuthError = (value: unknown): string => {
//   if (typeof value !== "object" || value === null) {
//     return "Doctor authentication account could not be created";
//   }

//   const data = value as Record<string, unknown>;

//   if (typeof data.message === "string" && data.message.trim()) {
//     return data.message;
//   }

//   if (typeof data.error === "object" && data.error !== null) {
//     const error = data.error as Record<string, unknown>;

//     if (typeof error.message === "string" && error.message.trim()) {
//       return error.message;
//     }
//   }

//   return "Doctor authentication account could not be created";
// };

// /* =========================================================
//    Admin patient management helpers
// ========================================================= */

// type ManagedPatientStatus = "active" | "blocked";

// const getAdminPatientFilter = (patientId: string): Filter<Document> => {
//   return {
//     $and: [getUserFilter(patientId), { role: "patient" }],
//   };
// };

// const formatManagedPatient = (patient: Document) => {
//   const status: ManagedPatientStatus =
//     patient.status === "blocked" ? "blocked" : "active";

//   return {
//     id: getDoctorDocumentId(patient),
//     name: getDoctorString(patient.name),
//     email: normalizeDoctorEmail(patient.email),
//     image: getDoctorString(patient.image) || null,
//     role: "patient" as const,
//     status,
//     emailVerified: patient.emailVerified === true,
//     phone: getDoctorString(patient.phone) || null,
//     address: getDoctorString(patient.address) || null,
//     dateOfBirth: getDoctorString(patient.dateOfBirth) || null,
//     gender: getDoctorString(patient.gender) || null,
//     bloodGroup: getDoctorString(patient.bloodGroup) || null,
//     occupation: getDoctorString(patient.occupation) || null,
//     city: getDoctorString(patient.city) || null,
//     country: getDoctorString(patient.country) || null,
//     bio: getDoctorString(patient.bio) || null,
//     emergencyContactName: getDoctorString(patient.emergencyContactName) || null,
//     emergencyContactPhone:
//       getDoctorString(patient.emergencyContactPhone) ||
//       getDoctorString(patient.emergencyContact) ||
//       null,
//     createdAt: formatDoctorDate(patient.createdAt),
//     updatedAt: formatDoctorDate(patient.updatedAt),
//   };
// };

// /* =========================================================
//    GET managed patients (10 per page)
// ========================================================= */

// app.get(
//   "/api/v1/admin/patients",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const search = getDoctorString(req.query.search);
//       const requestedStatus = getDoctorString(req.query.status);
//       const requestedPage = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;

//       const conditions: Filter<Document>[] = [{ role: "patient" }];

//       if (requestedStatus === "active" || requestedStatus === "blocked") {
//         conditions.push({ status: requestedStatus });
//       }

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         conditions.push({
//           $or: [
//             {
//               name: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               email: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       const filter: Filter<Document> = { $and: conditions };
//       const usersCollection = database.collection("user");
//       const total = await usersCollection.countDocuments(filter);
//       const totalPages = Math.max(1, Math.ceil(total / limit));
//       const page = Math.min(requestedPage, totalPages);

//       const patientDocuments = await usersCollection
//         .find(filter)
//         .sort({
//           updatedAt: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .skip((page - 1) * limit)
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         patients: patientDocuments.map(formatManagedPatient),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages,
//         },
//       });
//     } catch (error) {
//       console.error("Get managed patients error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patients",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET managed patient details
// ========================================================= */

// app.get(
//   "/api/v1/admin/patients/:patientId",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);

//       if (!patientId) {
//         res.status(400).json({
//           success: false,
//           message: "Patient ID is required",
//         });
//         return;
//       }

//       const patient = await database
//         .collection("user")
//         .findOne(getAdminPatientFilter(patientId));

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         patient: formatManagedPatient(patient),
//       });
//     } catch (error) {
//       console.error("Get managed patient details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient details",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH block or activate patient
// ========================================================= */

// app.patch(
//   "/api/v1/admin/patients/:patientId/status",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);
//       const requestedStatus = getDoctorString(req.body.status);

//       if (requestedStatus !== "active" && requestedStatus !== "blocked") {
//         res.status(400).json({
//           success: false,
//           message: "Status must be active or blocked",
//         });
//         return;
//       }

//       const usersCollection = database.collection("user");
//       const patient = await usersCollection.findOne(
//         getAdminPatientFilter(patientId),
//       );

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       const status = requestedStatus as ManagedPatientStatus;
//       const updatedPatient = await usersCollection.findOneAndUpdate(
//         { _id: patient._id },
//         {
//           $set: {
//             status,
//             updatedAt: new Date(),
//           },
//         },
//         { returnDocument: "after" },
//       );

//       if (!updatedPatient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       if (status === "blocked") {
//         await database.collection("session").deleteMany({
//           userId: getDoctorDocumentId(patient),
//         });
//       }

//       res.status(200).json({
//         success: true,
//         message:
//           status === "blocked"
//             ? "Patient blocked successfully"
//             : "Patient activated successfully",
//         patient: formatManagedPatient(updatedPatient),
//       });
//     } catch (error) {
//       console.error("Change patient status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to change patient status",
//       });
//     }
//   },
// );

// /* =========================================================
//    DELETE patient account
// ========================================================= */

// app.delete(
//   "/api/v1/admin/patients/:patientId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const patientId = getDoctorString(req.params.patientId);

//       if (!patientId) {
//         res.status(400).json({
//           success: false,
//           message: "Patient ID is required",
//         });
//         return;
//       }

//       const usersCollection = database.collection("user");
//       const patient = await usersCollection.findOne(
//         getAdminPatientFilter(patientId),
//       );

//       if (!patient) {
//         res.status(404).json({
//           success: false,
//           message: "Patient was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(patient);
//       const email = normalizeDoctorEmail(patient.email);

//       await Promise.all([
//         database.collection("session").deleteMany({ userId }),
//         database.collection("account").deleteMany({ userId }),
//         database.collection("verification").deleteMany({
//           $or: [{ identifier: email }, { value: email }],
//         }),
//       ]);

//       const deleteResult = await usersCollection.deleteOne({
//         _id: patient._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Patient account could not be deleted",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Patient account deleted successfully",
//         deletedPatientId: userId,
//       });
//     } catch (error) {
//       console.error("Delete patient account error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete patient account",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET all doctors
// ========================================================= */

// app.get(
//   "/api/v1/admin/doctors",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const search = getDoctorString(req.query.search);

//       const status = getDoctorString(req.query.status);

//       const page = Math.max(
//         1,
//         Math.floor(getDoctorNumber(req.query.page) || 1),
//       );

//       const limit = Math.min(
//         100,
//         Math.max(1, Math.floor(getDoctorNumber(req.query.limit) || 50)),
//       );

//       const filter: Filter<Document> = {};

//       if (status === "active" || status === "blocked") {
//         filter.status = status;
//       }

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         filter.$or = [
//           {
//             name: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             email: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             phone: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//           {
//             specialization: {
//               $regex: safeSearch,
//               $options: "i",
//             },
//           },
//         ];
//       }

//       const [doctorDocuments, total] = await Promise.all([
//         doctorsCollection
//           .find(filter)
//           .sort({
//             createdAt: -1,
//             _id: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         doctorsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,

//         doctors: doctorDocuments.map(formatDoctor),

//         pagination: {
//           page,
//           limit,
//           total,

//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get doctors error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    GET single doctor details
// ========================================================= */

// app.get(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       if (!doctorId) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         doctor: formatDoctor(doctor),
//       });
//     } catch (error) {
//       console.error("Get doctor details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor details",
//       });
//     }
//   },
// );

// /* =========================================================
//    POST create doctor
// ========================================================= */

// app.post(
//   "/api/v1/admin/doctors",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const name = getDoctorString(req.body.name);

//       const email = normalizeDoctorEmail(req.body.email);

//       const password = getDoctorString(req.body.password);

//       const specialization = getDoctorString(req.body.specialization);

//       if (!name || !email || !password || !specialization) {
//         res.status(400).json({
//           success: false,
//           message: "Name, email, password and specialization are required",
//         });

//         return;
//       }

//       if (!isValidDoctorEmail(email)) {
//         res.status(400).json({
//           success: false,
//           message: "A valid email address is required",
//         });

//         return;
//       }

//       if (password.length < 8) {
//         res.status(400).json({
//           success: false,
//           message: "Password must contain at least 8 characters",
//         });

//         return;
//       }

//       const usersCollection = database.collection("user");

//       const doctorsCollection = database.collection("doctors");

//       const accountsCollection = database.collection("account");

//       const sessionsCollection = database.collection("session");

//       const existingUser = await usersCollection.findOne({
//         email,
//       });

//       const existingDoctor = await doctorsCollection.findOne({
//         email,
//       });

//       if (existingUser || existingDoctor) {
//         res.status(409).json({
//           success: false,
//           message: "An account with this email already exists",
//         });

//         return;
//       }

//       /*
//        * Better Auth securely creates the email/password account.
//        * Raw password MongoDB-তে save হবে না।
//        */
//       const signupResponse = await fetch(
//         `${betterAuthUrl}/api/auth/sign-up/email`,
//         {
//           method: "POST",

//           headers: {
//             "content-type": "application/json",
//             accept: "application/json",
//             origin: betterAuthUrl,
//           },

//           body: JSON.stringify({
//             name,
//             email,
//             password,
//           }),
//         },
//       );

//       const signupData = await readBetterAuthResponse(signupResponse);

//       if (!signupResponse.ok) {
//         res
//           .status(signupResponse.status >= 500 ? 502 : signupResponse.status)
//           .json({
//             success: false,
//             message: getBetterAuthError(signupData),
//           });

//         return;
//       }

//       const createdUser = await usersCollection.findOne({
//         email,
//       });

//       if (!createdUser) {
//         res.status(500).json({
//           success: false,
//           message: "Authentication account was created but user was not found",
//         });

//         return;
//       }

//       const userId = getDoctorDocumentId(createdUser);

//       const now = new Date();

//       await usersCollection.updateOne(
//         {
//           _id: createdUser._id,
//         },
//         {
//           $set: {
//             name,
//             email,
//             role: "doctor",
//             status: "active",
//             updatedAt: now,
//           },
//         },
//       );

//       /*
//        * Admin-created doctor will sign in manually.
//        * Remove any session created by signup.
//        */
//       await sessionsCollection.deleteMany({
//         userId,
//       });

//       const doctorDocument = {
//         userId,

//         name,
//         email,

//         image: getDoctorString(req.body.image) || null,

//         phone: getDoctorString(req.body.phone),

//         specialization,

//         qualification: getDoctorString(req.body.qualification),

//         experienceYears: getDoctorNumber(req.body.experienceYears),

//         hospital: getDoctorString(req.body.hospital),

//         address: getDoctorString(req.body.address),

//         bio: getDoctorString(req.body.bio),

//         role: "doctor" as const,

//         status: "active" as const,

//         createdAt: now,
//         updatedAt: now,
//       };

//       try {
//         const insertResult = await doctorsCollection.insertOne(doctorDocument);

//         const createdDoctor = await doctorsCollection.findOne({
//           _id: insertResult.insertedId,
//         });

//         if (!createdDoctor) {
//           throw new Error("Created doctor profile was not found");
//         }

//         res.status(201).json({
//           success: true,
//           message: "Doctor created successfully",
//           doctor: formatDoctor(createdDoctor),
//         });
//       } catch (profileError) {
//         /*
//          * Roll back authentication data if
//          * doctor profile creation fails.
//          */
//         await Promise.all([
//           sessionsCollection.deleteMany({
//             userId,
//           }),

//           accountsCollection.deleteMany({
//             userId,
//           }),

//           usersCollection.deleteOne({
//             _id: createdUser._id,
//           }),
//         ]);

//         throw profileError;
//       }
//     } catch (error) {
//       console.error("Create doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to create doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH edit doctor
// ========================================================= */

// app.patch(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const name = getDoctorString(req.body.name);

//       const email = normalizeDoctorEmail(req.body.email);

//       const specialization = getDoctorString(req.body.specialization);

//       if (!doctorId || !name || !email || !specialization) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID, name, email and specialization are required",
//         });

//         return;
//       }

//       if (!isValidDoctorEmail(email)) {
//         res.status(400).json({
//           success: false,
//           message: "A valid email address is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       const linkedUser = userId
//         ? await usersCollection.findOne(getUserFilter(userId))
//         : null;

//       const duplicateDoctor = await doctorsCollection.findOne({
//         email,

//         _id: {
//           $ne: doctor._id,
//         },
//       });

//       const duplicateUser = await usersCollection.findOne({
//         email,

//         ...(linkedUser
//           ? {
//               _id: {
//                 $ne: linkedUser._id,
//               },
//             }
//           : {}),
//       });

//       if (duplicateDoctor || duplicateUser) {
//         res.status(409).json({
//           success: false,
//           message: "Another account already uses this email",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedDoctor = await doctorsCollection.findOneAndUpdate(
//         {
//           _id: doctor._id,
//         },
//         {
//           $set: {
//             name,
//             email,

//             image: getDoctorString(req.body.image) || null,

//             phone: getDoctorString(req.body.phone),

//             specialization,

//             qualification: getDoctorString(req.body.qualification),

//             experienceYears: getDoctorNumber(req.body.experienceYears),

//             hospital: getDoctorString(req.body.hospital),

//             address: getDoctorString(req.body.address),

//             bio: getDoctorString(req.body.bio),

//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       if (!updatedDoctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       if (linkedUser) {
//         await usersCollection.updateOne(
//           {
//             _id: linkedUser._id,
//           },
//           {
//             $set: {
//               name,
//               email,

//               image: getDoctorString(req.body.image) || null,

//               updatedAt: now,
//             },
//           },
//         );
//       }

//       res.status(200).json({
//         success: true,
//         message: "Doctor updated successfully",
//         doctor: formatDoctor(updatedDoctor),
//       });
//     } catch (error) {
//       console.error("Update doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    PATCH block or activate doctor
// ========================================================= */

// app.patch(
//   "/api/v1/admin/doctors/:doctorId/status",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const requestedStatus = getDoctorString(req.body.status);

//       if (requestedStatus !== "active" && requestedStatus !== "blocked") {
//         res.status(400).json({
//           success: false,
//           message: "Status must be active or blocked",
//         });

//         return;
//       }

//       const status = requestedStatus as DoctorStatus;

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedDoctor = await doctorsCollection.findOneAndUpdate(
//         {
//           _id: doctor._id,
//         },
//         {
//           $set: {
//             status,
//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       if (!updatedDoctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       if (userId) {
//         await usersCollection.updateOne(getUserFilter(userId), {
//           $set: {
//             status,
//             updatedAt: now,
//           },
//         });
//       }

//       res.status(200).json({
//         success: true,

//         message:
//           status === "blocked"
//             ? "Doctor blocked successfully"
//             : "Doctor activated successfully",

//         doctor: formatDoctor(updatedDoctor),
//       });
//     } catch (error) {
//       console.error("Change doctor status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to change doctor status",
//       });
//     }
//   },
// );

// /* =========================================================
//    DELETE doctor
// ========================================================= */

// app.delete(
//   "/api/v1/admin/doctors/:doctorId",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       if (!doctorId) {
//         res.status(400).json({
//           success: false,
//           message: "Doctor ID is required",
//         });

//         return;
//       }

//       const doctorsCollection = database.collection("doctors");

//       const usersCollection = database.collection("user");

//       const accountsCollection = database.collection("account");

//       const sessionsCollection = database.collection("session");

//       const verificationCollection = database.collection("verification");

//       const doctor = await doctorsCollection.findOne(getDoctorFilter(doctorId));

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const userId = getDoctorString(doctor.userId);

//       if (userId) {
//         await Promise.all([
//           sessionsCollection.deleteMany({
//             userId,
//           }),

//           accountsCollection.deleteMany({
//             userId,
//           }),

//           verificationCollection.deleteMany({
//             $or: [
//               {
//                 identifier: doctor.email,
//               },
//               {
//                 value: doctor.email,
//               },
//             ],
//           }),
//         ]);

//         await usersCollection.deleteOne(getUserFilter(userId));
//       }

//       const deleteResult = await doctorsCollection.deleteOne({
//         _id: doctor._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Doctor could not be deleted",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Doctor deleted successfully",
//         deletedDoctorId: getDoctorDocumentId(doctor),
//       });
//     } catch (error) {
//       console.error("Delete doctor error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete doctor",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctors, appointments and reviews
// ========================================================= */

// type AppointmentStatus = "pending" | "approved" | "completed" | "rejected";

// const ACTIVE_APPOINTMENT_STATUSES: AppointmentStatus[] = [
//   "pending",
//   "approved",
// ];

// const getPositiveInteger = (
//   value: unknown,
//   fallback: number,
//   maximum: number,
// ): number => {
//   const parsed = Number(value);

//   if (!Number.isFinite(parsed)) {
//     return fallback;
//   }

//   return Math.min(maximum, Math.max(1, Math.floor(parsed)));
// };

// const getCurrentDatabaseUser = async (
//   req: AuthenticatedRequest,
// ): Promise<Document | null> => {
//   if (!database || !req.userId) {
//     return null;
//   }

//   const conditions: Filter<Document>[] = [getUserFilter(req.userId)];

//   if (req.userEmail) {
//     conditions.push({
//       email: req.userEmail.toLowerCase(),
//     });
//   }

//   return database.collection("user").findOne({
//     $or: conditions,
//   });
// };

// const getNormalizedUserRole = (user: Document): UserRole => {
//   return user.role === "admin" ||
//     user.role === "doctor" ||
//     user.role === "patient"
//     ? user.role
//     : "patient";
// };

// const getNormalizedUserStatus = (user: Document): UserStatus => {
//   return user.status === "blocked" ? "blocked" : "active";
// };

// const getPublicDoctor = (doctor: Document) => {
//   const ratingAverage = Number(doctor.ratingAverage);

//   const ratingCount = Number(doctor.ratingCount);

//   return {
//     ...formatDoctor(doctor),

//     ratingAverage: Number.isFinite(ratingAverage)
//       ? Number(ratingAverage.toFixed(1))
//       : 0,

//     ratingCount: Number.isFinite(ratingCount)
//       ? Math.max(0, Math.floor(ratingCount))
//       : 0,
//   };
// };

// const getReviewDocumentId = (document: Document): string => {
//   return getDoctorDocumentId(document);
// };

// const formatReview = (review: Document) => {
//   return {
//     id: getReviewDocumentId(review),
//     doctorId: getDoctorString(review.doctorId),
//     userId: getDoctorString(review.userId),
//     userName: getDoctorString(review.userName),
//     userEmail: normalizeDoctorEmail(review.userEmail),
//     userImage: getDoctorString(review.userImage) || null,
//     rating: Math.min(
//       5,
//       Math.max(1, Math.floor(getDoctorNumber(review.rating))),
//     ),
//     review: getDoctorString(review.review),
//     createdAt: formatDoctorDate(review.createdAt),
//     updatedAt: formatDoctorDate(review.updatedAt),
//   };
// };

// const refreshDoctorRatingStats = async (doctorId: string): Promise<void> => {
//   if (!database) {
//     return;
//   }

//   const reviewsCollection = database.collection("reviews");

//   const doctorsCollection = database.collection("doctors");

//   const [stats] = await reviewsCollection
//     .aggregate([
//       {
//         $match: {
//           doctorId,
//         },
//       },
//       {
//         $group: {
//           _id: "$doctorId",
//           ratingAverage: {
//             $avg: "$rating",
//           },
//           ratingCount: {
//             $sum: 1,
//           },
//         },
//       },
//     ])
//     .toArray();

//   await doctorsCollection.updateOne(getDoctorFilter(doctorId), {
//     $set: {
//       ratingAverage:
//         typeof stats?.ratingAverage === "number"
//           ? Number(stats.ratingAverage.toFixed(2))
//           : 0,
//       ratingCount:
//         typeof stats?.ratingCount === "number" ? stats.ratingCount : 0,
//       updatedAt: new Date(),
//     },
//   });
// };

// const formatAppointment = (appointment: Document) => {
//   return {
//     id: getDoctorDocumentId(appointment),
//     doctorId: getDoctorString(appointment.doctorId),
//     doctorUserId: getDoctorString(appointment.doctorUserId),
//     doctorName: getDoctorString(appointment.doctorName),
//     doctorImage: getDoctorString(appointment.doctorImage) || null,
//     specialization: getDoctorString(appointment.specialization),
//     hospital: getDoctorString(appointment.hospital),
//     patientUserId: getDoctorString(appointment.patientUserId),
//     patientName: getDoctorString(appointment.patientName),
//     patientEmail: normalizeDoctorEmail(appointment.patientEmail),
//     patientImage: getDoctorString(appointment.patientImage) || null,
//     phone: getDoctorString(appointment.phone),
//     address: getDoctorString(appointment.address),
//     problemTitle: getDoctorString(appointment.problemTitle),
//     symptomsDescription: getDoctorString(appointment.symptomsDescription),
//     appointmentDate: getDoctorString(appointment.appointmentDate),
//     appointmentTime: getDoctorString(appointment.appointmentTime),
//     status:
//       appointment.status === "approved" ||
//       appointment.status === "completed" ||
//       appointment.status === "rejected"
//         ? appointment.status
//         : "pending",
//     rejectionReason: getDoctorString(appointment.rejectionReason) || null,
//     approvedAt: formatDoctorDate(appointment.approvedAt),
//     completedAt: formatDoctorDate(appointment.completedAt),
//     rejectedAt: formatDoctorDate(appointment.rejectedAt),
//     rescheduledAt: formatDoctorDate(appointment.rescheduledAt),
//     rescheduledBy: getDoctorString(appointment.rescheduledBy) || null,
//     rescheduleReason: getDoctorString(appointment.rescheduleReason) || null,
//     createdAt: formatDoctorDate(appointment.createdAt),
//     updatedAt: formatDoctorDate(appointment.updatedAt),
//   };
// };

// const attachPatientImages = async (
//   appointments: Document[],
// ): Promise<Document[]> => {
//   if (!database || appointments.length === 0) {
//     return appointments;
//   }

//   const patientUserIds = Array.from(
//     new Set(
//       appointments
//         .map((appointment) => getDoctorString(appointment.patientUserId))
//         .filter(Boolean),
//     ),
//   );

//   if (patientUserIds.length === 0) {
//     return appointments;
//   }

//   const objectIds = patientUserIds
//     .filter((userId) => ObjectId.isValid(userId))
//     .map((userId) => new ObjectId(userId));

//   const userConditions: Filter<Document>[] = [
//     {
//       id: {
//         $in: patientUserIds,
//       },
//     },
//   ];

//   if (objectIds.length > 0) {
//     userConditions.push({
//       _id: {
//         $in: objectIds,
//       },
//     });
//   }

//   const users = await database
//     .collection("user")
//     .find(
//       {
//         $or: userConditions,
//       },
//       {
//         projection: {
//           id: 1,
//           image: 1,
//         },
//       },
//     )
//     .toArray();

//   const imageByUserId = new Map<string, string | null>();

//   users.forEach((user) => {
//     imageByUserId.set(
//       getDoctorDocumentId(user),
//       getDoctorString(user.image) || null,
//     );
//   });

//   return appointments.map((appointment) => ({
//     ...appointment,
//     patientImage:
//       getDoctorString(appointment.patientImage) ||
//       imageByUserId.get(getDoctorString(appointment.patientUserId)) ||
//       null,
//   }));
// };

// /* =========================================================
//    Public doctor filters
// ========================================================= */

// app.get(
//   "/api/v1/doctors/filters",
//   async (_req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorDocuments = await database
//         .collection("doctors")
//         .find(
//           {
//             status: "active",
//           },
//           {
//             projection: {
//               specialization: 1,
//               qualification: 1,
//               experienceYears: 1,
//               hospital: 1,
//               chamber: 1,
//             },
//           },
//         )
//         .toArray();

//       const specializations = new Set<string>();
//       const qualifications = new Set<string>();
//       const hospitals = new Set<string>();
//       const experienceYears = new Set<number>();

//       doctorDocuments.forEach((doctor) => {
//         const specialization = getDoctorString(doctor.specialization);
//         const qualification = getDoctorString(doctor.qualification);
//         const hospital =
//           getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber);
//         const experience = getDoctorNumber(doctor.experienceYears);

//         if (specialization) {
//           specializations.add(specialization);
//         }

//         if (qualification) {
//           qualifications.add(qualification);
//         }

//         if (hospital) {
//           hospitals.add(hospital);
//         }

//         experienceYears.add(experience);
//       });

//       res.status(200).json({
//         success: true,
//         filters: {
//           specializations: Array.from(specializations).sort((a, b) =>
//             a.localeCompare(b),
//           ),
//           qualifications: Array.from(qualifications).sort((a, b) =>
//             a.localeCompare(b),
//           ),
//           hospitals: Array.from(hospitals).sort((a, b) => a.localeCompare(b)),
//           experienceYears: Array.from(experienceYears).sort((a, b) => a - b),
//         },
//       });
//     } catch (error) {
//       console.error("Get public doctor filters error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor filters",
//       });
//     }
//   },
// );

// /* =========================================================
//    Top Rated Public Doctors
// ========================================================= */

// app.get(
//   "/api/v1/doctors/top-rated",
//   async (_req: Request, res: Response): Promise<void> => {
//     try {

//       if (!database) {
//         res.status(503).json({
//           success:false,
//           message:"Database is not connected",
//         });

//         return;
//       }

//       const doctorsCollection =
//         database.collection("doctors");

//       const doctors =
//         await doctorsCollection
//           .find({
//             status:"active",
//           })
//           .sort({
//             ratingAverage:-1,
//             ratingCount:-1,
//             createdAt:-1,
//             _id:-1,
//           })
//           .limit(4)
//           .toArray();

//       res.status(200).json({

//         success:true,

//         doctors:
//           doctors.map(getPublicDoctor),

//       });

//     } catch(error){

//       console.error(
//         "Get top rated doctors error:",
//         error
//       );

//       res.status(500).json({

//         success:false,

//         message:
//         "Failed to retrieve top rated doctors",

//       });

//     }
//   },
// );

// /* =========================================================
//    Public doctor list
// ========================================================= */

// app.get(
//   "/api/v1/doctors",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const search = getDoctorString(req.query.search);
//       const specialization = getDoctorString(req.query.specialization);
//       const qualification = getDoctorString(req.query.qualification);
//       const hospital = getDoctorString(req.query.hospital);
//       const experienceValue = getDoctorString(req.query.experienceYears);

//       const page = getPositiveInteger(req.query.page, 1, 100000);

//       const limit = 8;

//       const conditions: Filter<Document>[] = [
//         {
//           status: "active",
//         },
//       ];

//       if (search) {
//         const safeSearch = escapeDoctorSearch(search);

//         conditions.push({
//           $or: [
//             {
//               name: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               specialization: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//             {
//               qualification: {
//                 $regex: safeSearch,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       if (specialization) {
//         conditions.push({
//           specialization: {
//             $regex: `^${escapeDoctorSearch(specialization)}$`,
//             $options: "i",
//           },
//         });
//       }

//       if (qualification) {
//         conditions.push({
//           qualification: {
//             $regex: `^${escapeDoctorSearch(qualification)}$`,
//             $options: "i",
//           },
//         });
//       }

//       if (hospital) {
//         const safeHospital = `^${escapeDoctorSearch(hospital)}$`;

//         conditions.push({
//           $or: [
//             {
//               hospital: {
//                 $regex: safeHospital,
//                 $options: "i",
//               },
//             },
//             {
//               chamber: {
//                 $regex: safeHospital,
//                 $options: "i",
//               },
//             },
//           ],
//         });
//       }

//       if (experienceValue) {
//         const experienceYears = Number(experienceValue);

//         if (Number.isFinite(experienceYears)) {
//           conditions.push({
//             experienceYears: Math.max(0, Math.floor(experienceYears)),
//           });
//         }
//       }

//       const filter: Filter<Document> = {
//         $and: conditions,
//       };

//       const doctorsCollection = database.collection("doctors");

//       const [doctorDocuments, total] = await Promise.all([
//         doctorsCollection
//           .find(filter)
//           .sort({
//             ratingAverage: -1,
//             ratingCount: -1,
//             createdAt: -1,
//             _id: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         doctorsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         doctors: doctorDocuments.map(getPublicDoctor),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get public doctors error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve public doctors",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public single doctor details
// ========================================================= */

// app.get(
//   "/api/v1/doctors/:doctorId",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         doctor: getPublicDoctor(doctor),
//       });
//     } catch (error) {
//       console.error("Get public doctor details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Public doctor reviews
// ========================================================= */

// app.get(
//   "/api/v1/doctors/:doctorId/reviews",
//   async (req: Request, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = getPositiveInteger(req.query.limit, 10, 50);

//       const reviewsCollection = database.collection("reviews");

//       const [reviewDocuments, total] = await Promise.all([
//         reviewsCollection
//           .find({
//             doctorId,
//           })
//           .sort({
//             updatedAt: -1,
//             createdAt: -1,
//           })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),

//         reviewsCollection.countDocuments({
//           doctorId,
//         }),
//       ]);

//       res.status(200).json({
//         success: true,
//         reviews: reviewDocuments.map(formatReview),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get doctor reviews error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve doctor reviews",
//       });
//     }
//   },
// );

// /* =========================================================
//    Create doctor review
// ========================================================= */

// app.post(
//   "/api/v1/doctors/:doctorId/reviews",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot submit a rating or review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const rating = Math.floor(Number(req.body.rating));
//       const reviewText = getDoctorString(req.body.review);

//       if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
//         res.status(400).json({
//           success: false,
//           message: "Rating must be a number from 1 to 5",
//         });

//         return;
//       }

//       if (reviewText.length > 2000) {
//         res.status(400).json({
//           success: false,
//           message: "Review cannot contain more than 2000 characters",
//         });

//         return;
//       }

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(doctor.userId) === currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "A doctor cannot review their own profile",
//         });

//         return;
//       }

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         doctorId,
//         userId: currentUserId,
//       });

//       if (existingReview) {
//         res.status(409).json({
//           success: false,
//           message:
//             "You have already reviewed this doctor. Please edit your existing review.",
//           code: "REVIEW_ALREADY_EXISTS",
//         });

//         return;
//       }

//       const now = new Date();

//       const reviewDocument = {
//         doctorId,
//         doctorUserId: getDoctorString(doctor.userId),
//         userId: currentUserId,
//         userName: getDoctorString(currentUser.name),
//         userEmail: normalizeDoctorEmail(currentUser.email),
//         userImage: getDoctorString(currentUser.image) || null,
//         rating,
//         review: reviewText,
//         createdAt: now,
//         updatedAt: now,
//       };

//       const insertResult = await reviewsCollection.insertOne(reviewDocument);

//       await refreshDoctorRatingStats(doctorId);

//       const createdReview = await reviewsCollection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "Rating and review submitted successfully",
//         review: createdReview ? formatReview(createdReview) : null,
//       });
//     } catch (error) {
//       console.error("Create doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to submit rating and review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Update doctor review
// ========================================================= */

// app.patch(
//   "/api/v1/doctors/:doctorId/reviews/:reviewId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot edit a review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const reviewId = getDoctorString(req.params.reviewId);
//       const rating = Math.floor(Number(req.body.rating));
//       const reviewText = getDoctorString(req.body.review);

//       if (!Number.isFinite(rating) || rating < 1 || rating > 5) {
//         res.status(400).json({
//           success: false,
//           message: "Rating must be a number from 1 to 5",
//         });

//         return;
//       }

//       if (reviewText.length > 2000) {
//         res.status(400).json({
//           success: false,
//           message: "Review cannot contain more than 2000 characters",
//         });

//         return;
//       }

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         $and: [
//           getDoctorFilter(reviewId),
//           {
//             doctorId,
//           },
//         ],
//       });

//       if (!existingReview) {
//         res.status(404).json({
//           success: false,
//           message: "Review was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(existingReview.userId) !== currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can edit only your own review",
//         });

//         return;
//       }

//       const updatedReview = await reviewsCollection.findOneAndUpdate(
//         {
//           _id: existingReview._id,
//         },
//         {
//           $set: {
//             rating,
//             review: reviewText,
//             updatedAt: new Date(),
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       await refreshDoctorRatingStats(doctorId);

//       res.status(200).json({
//         success: true,
//         message: "Rating and review updated successfully",
//         review: updatedReview ? formatReview(updatedReview) : null,
//       });
//     } catch (error) {
//       console.error("Update doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update rating and review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Delete doctor review
// ========================================================= */

// app.delete(
//   "/api/v1/doctors/:doctorId/reviews/:reviewId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       if (getNormalizedUserStatus(currentUser) === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot delete a review.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);
//       const reviewId = getDoctorString(req.params.reviewId);

//       const reviewsCollection = database.collection("reviews");

//       const existingReview = await reviewsCollection.findOne({
//         $and: [
//           getDoctorFilter(reviewId),
//           {
//             doctorId,
//           },
//         ],
//       });

//       if (!existingReview) {
//         res.status(404).json({
//           success: false,
//           message: "Review was not found",
//         });

//         return;
//       }

//       const currentUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(existingReview.userId) !== currentUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can delete only your own review",
//         });

//         return;
//       }

//       await reviewsCollection.deleteOne({
//         _id: existingReview._id,
//       });

//       await refreshDoctorRatingStats(doctorId);

//       res.status(200).json({
//         success: true,
//         message: "Review deleted successfully",
//         deletedReviewId: reviewId,
//       });
//     } catch (error) {
//       console.error("Delete doctor review error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete review",
//       });
//     }
//   },
// );

// /* =========================================================
//    Appointment eligibility
// ========================================================= */

// app.get(
//   "/api/v1/appointments/eligibility/:doctorId",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);

//       if (role !== "patient") {
//         res.status(403).json({
//           success: false,
//           canBook: false,
//           code: "PATIENT_ONLY",
//           message: "Only patients can take a doctor appointment.",
//         });

//         return;
//       }

//       if (status === "blocked") {
//         res.status(403).json({
//           success: false,
//           canBook: false,
//           code: "ACCOUNT_BLOCKED",
//           message:
//             "You are restricted by the administrator and cannot take an appointment.",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.params.doctorId);

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           canBook: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);

//       const existingAppointment = await database
//         .collection("appointments")
//         .findOne({
//           doctorId,
//           patientUserId,
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//       if (existingAppointment) {
//         res.status(200).json({
//           success: true,
//           canBook: false,
//           code: "APPOINTMENT_ALREADY_EXISTS",
//           message:
//             "You already have a pending or approved appointment with this doctor.",
//           appointment: formatAppointment(existingAppointment),
//         });

//         return;
//       }

//       res.status(200).json({
//         success: true,
//         canBook: true,
//         message: "You can take an appointment with this doctor.",
//       });
//     } catch (error) {
//       console.error("Appointment eligibility error:", error);

//       res.status(500).json({
//         success: false,
//         canBook: false,
//         message: "Failed to check appointment eligibility",
//       });
//     }
//   },
// );

// /* =========================================================
//    Create appointment
// ========================================================= */

// app.post(
//   "/api/v1/appointments",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);

//       if (role !== "patient") {
//         res.status(403).json({
//           success: false,
//           message: "Only patients can take a doctor appointment.",
//           code: "PATIENT_ONLY",
//         });

//         return;
//       }

//       if (status === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "You are restricted by the administrator and cannot take an appointment.",
//           code: "ACCOUNT_BLOCKED",
//         });

//         return;
//       }

//       const doctorId = getDoctorString(req.body.doctorId);
//       const patientName = getDoctorString(req.body.patientName);
//       const phone = getDoctorString(req.body.phone);
//       const address = getDoctorString(req.body.address);
//       const problemTitle = getDoctorString(req.body.problemTitle);
//       const symptomsDescription = getDoctorString(req.body.symptomsDescription);
//       const appointmentDate = getDoctorString(req.body.appointmentDate);
//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       if (
//         !doctorId ||
//         !patientName ||
//         !phone ||
//         !address ||
//         !problemTitle ||
//         !symptomsDescription ||
//         !appointmentDate ||
//         !appointmentTime
//       ) {
//         res.status(400).json({
//           success: false,
//           message:
//             "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
//         });

//         return;
//       }

//       if (symptomsDescription.length > 5000) {
//         res.status(400).json({
//           success: false,
//           message:
//             "Symptoms description cannot contain more than 5000 characters",
//         });

//         return;
//       }

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       const doctor = await database.collection("doctors").findOne({
//         $and: [
//           getDoctorFilter(doctorId),
//           {
//             status: "active",
//           },
//         ],
//       });

//       if (!doctor) {
//         res.status(404).json({
//           success: false,
//           message: "Doctor was not found",
//         });

//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentsCollection = database.collection("appointments");

//       const existingAppointment = await appointmentsCollection.findOne({
//         doctorId,
//         patientUserId,
//         status: {
//           $in: ACTIVE_APPOINTMENT_STATUSES,
//         },
//       });

//       if (existingAppointment) {
//         res.status(409).json({
//           success: false,
//           message:
//             "You already have a pending or approved appointment with this doctor.",
//           code: "APPOINTMENT_ALREADY_EXISTS",
//           appointment: formatAppointment(existingAppointment),
//         });

//         return;
//       }

//       const now = new Date();

//       const appointmentDocument = {
//         doctorId,
//         doctorUserId: getDoctorString(doctor.userId),
//         doctorName: getDoctorString(doctor.name),
//         doctorImage: getDoctorString(doctor.image) || null,
//         specialization: getDoctorString(doctor.specialization),
//         hospital:
//           getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
//         patientUserId,
//         patientName,
//         patientEmail: normalizeDoctorEmail(currentUser.email),
//         patientImage: getDoctorString(currentUser.image) || null,
//         phone,
//         address,
//         problemTitle,
//         symptomsDescription,
//         appointmentDate,
//         appointmentTime,
//         status: "pending" as const,
//         createdAt: now,
//         updatedAt: now,
//       };

//       const insertResult =
//         await appointmentsCollection.insertOne(appointmentDocument);

//       const createdAppointment = await appointmentsCollection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "Appointment request submitted successfully",
//         appointment: createdAppointment
//           ? formatAppointment(createdAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Create appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to submit appointment request",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient appointment helpers
// ========================================================= */

// const getPatientAppointment = async (
//   patientUserId: string,
//   appointmentId: string,
// ): Promise<Document | null> => {
//   if (!database) {
//     return null;
//   }

//   return database.collection("appointments").findOne({
//     $and: [
//       getDoctorFilter(appointmentId),
//       {
//         patientUserId,
//       },
//     ],
//   });
// };

// const getAppointmentDoctor = async (
//   appointment: Document,
// ): Promise<Document | null> => {
//   if (!database) {
//     return null;
//   }

//   const doctorId = getDoctorString(appointment.doctorId);

//   if (!doctorId) {
//     return null;
//   }

//   return database.collection("doctors").findOne(getDoctorFilter(doctorId));
// };

// const validatePatientAppointmentInput = (
//   body: Record<string, unknown>,
// ):
//   | {
//       success: true;
//       values: {
//         patientName: string;
//         phone: string;
//         address: string;
//         problemTitle: string;
//         symptomsDescription: string;
//         appointmentDate: string;
//         appointmentTime: string;
//       };
//     }
//   | {
//       success: false;
//       message: string;
//     } => {
//   const patientName = getDoctorString(body.patientName);
//   const phone = getDoctorString(body.phone);
//   const address = getDoctorString(body.address);
//   const problemTitle = getDoctorString(body.problemTitle);
//   const symptomsDescription = getDoctorString(body.symptomsDescription);
//   const appointmentDate = getDoctorString(body.appointmentDate);
//   const appointmentTime = getDoctorString(body.appointmentTime);

//   if (
//     !patientName ||
//     !phone ||
//     !address ||
//     !problemTitle ||
//     !symptomsDescription ||
//     !appointmentDate ||
//     !appointmentTime
//   ) {
//     return {
//       success: false,
//       message:
//         "Patient name, phone, address, health problem title, symptoms description, appointment date and time are required.",
//     };
//   }

//   if (patientName.length > 150) {
//     return {
//       success: false,
//       message: "Patient name cannot contain more than 150 characters",
//     };
//   }

//   if (phone.length > 40) {
//     return {
//       success: false,
//       message: "Phone number cannot contain more than 40 characters",
//     };
//   }

//   if (address.length > 500) {
//     return {
//       success: false,
//       message: "Address cannot contain more than 500 characters",
//     };
//   }

//   if (problemTitle.length > 250) {
//     return {
//       success: false,
//       message: "Health problem title cannot contain more than 250 characters",
//     };
//   }

//   if (symptomsDescription.length > 5000) {
//     return {
//       success: false,
//       message: "Symptoms description cannot contain more than 5000 characters",
//     };
//   }

//   const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//   const timePattern = /^\d{2}:\d{2}$/;

//   if (
//     !datePattern.test(appointmentDate) ||
//     !timePattern.test(appointmentTime)
//   ) {
//     return {
//       success: false,
//       message: "A valid appointment date and time are required",
//     };
//   }

//   const today = new Date().toISOString().slice(0, 10);

//   if (appointmentDate < today) {
//     return {
//       success: false,
//       message: "Appointment date cannot be in the past",
//     };
//   }

//   return {
//     success: true,
//     values: {
//       patientName,
//       phone,
//       address,
//       problemTitle,
//       symptomsDescription,
//       appointmentDate,
//       appointmentTime,
//     },
//   };
// };

// /* =========================================================
//    Patient appointments list
// ========================================================= */

// app.get(
//   "/api/v1/patient/appointments",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;
//       const appointmentsCollection = database.collection("appointments");
//       const filter: Filter<Document> = { patientUserId };

//       const [appointmentDocuments, total] = await Promise.all([
//         appointmentsCollection
//           .find(filter)
//           .sort({ createdAt: -1, _id: -1 })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),
//         appointmentsCollection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         appointments: appointmentDocuments.map(formatAppointment),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get patient appointments error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient single appointment details
// ========================================================= */

// app.get(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);

//       if (!appointmentId) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment ID is required",
//         });
//         return;
//       }

//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       const doctor = await getAppointmentDoctor(appointment);

//       res.status(200).json({
//         success: true,
//         appointment: formatAppointment(appointment),
//         doctor: doctor ? getPublicDoctor(doctor) : null,
//       });
//     } catch (error) {
//       console.error("Get patient appointment details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointment details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient edit appointment
// ========================================================= */

// app.patch(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus !== "pending" && currentStatus !== "rejected") {
//         res.status(409).json({
//           success: false,
//           message: "Only pending or rejected appointments can be edited",
//         });
//         return;
//       }

//       const validation = validatePatientAppointmentInput(
//         req.body as Record<string, unknown>,
//       );

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       if (currentStatus === "rejected") {
//         const anotherActiveAppointment = await database
//           .collection("appointments")
//           .findOne({
//             _id: { $ne: appointment._id },
//             doctorId: getDoctorString(appointment.doctorId),
//             patientUserId,
//             status: { $in: ACTIVE_APPOINTMENT_STATUSES },
//           });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "You already have another pending or approved appointment with this doctor.",
//           });
//           return;
//         }
//       }

//       const now = new Date();
//       const updatedAppointment = await database
//         .collection("appointments")
//         .findOneAndUpdate(
//           { _id: appointment._id },
//           {
//             $set: {
//               ...validation.values,
//               status: "pending",
//               rejectionReason: null,
//               rejectedAt: null,
//               approvedAt: null,
//               completedAt: null,
//               updatedAt: now,
//             },
//           },
//           { returnDocument: "after" },
//         );

//       res.status(200).json({
//         success: true,
//         message:
//           currentStatus === "rejected"
//             ? "Appointment updated and resubmitted successfully"
//             : "Appointment updated successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update patient appointment error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient cancel and delete appointment
// ========================================================= */

// app.delete(
//   "/api/v1/patient/appointments/:appointmentId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointment = await getPatientAppointment(
//         patientUserId,
//         appointmentId,
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       if (getDoctorString(appointment.status) === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be cancelled or deleted",
//         });
//         return;
//       }

//       const deleteResult = await database
//         .collection("appointments")
//         .deleteOne({ _id: appointment._id });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "Appointment could not be cancelled",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         message: "Appointment cancelled and removed successfully",
//         deletedAppointmentId: appointmentId,
//       });
//     } catch (error) {
//       console.error("Cancel patient appointment error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to cancel appointment",
//       });
//     }
//   },
// );

// const getAppointmentListFilter = (
//   req: AuthenticatedRequest,
// ): Filter<Document> => {
//   const conditions: Filter<Document>[] = [];
//   const status = getDoctorString(req.query.status);
//   const search = getDoctorString(req.query.search);

//   if (
//     status === "pending" ||
//     status === "approved" ||
//     status === "completed" ||
//     status === "rejected"
//   ) {
//     conditions.push({
//       status,
//     });
//   }

//   if (search) {
//     const safeSearch = escapeDoctorSearch(search);

//     conditions.push({
//       $or: [
//         {
//           patientName: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           patientEmail: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           doctorName: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//         {
//           problemTitle: {
//             $regex: safeSearch,
//             $options: "i",
//           },
//         },
//       ],
//     });
//   }

//   return conditions.length
//     ? {
//         $and: conditions,
//       }
//     : {};
// };

// const sendAppointmentList = async (
//   req: AuthenticatedRequest,
//   res: Response,
//   additionalFilter: Filter<Document> = {},
// ): Promise<void> => {
//   if (!database) {
//     res.status(503).json({
//       success: false,
//       message: "Database is not connected",
//     });

//     return;
//   }

//   const page = getPositiveInteger(req.query.page, 1, 100000);
//   const limit = 10;

//   const queryFilter = getAppointmentListFilter(req);

//   const filter: Filter<Document> = {
//     $and: [queryFilter, additionalFilter],
//   };

//   const appointmentsCollection = database.collection("appointments");

//   const [appointmentDocuments, total] = await Promise.all([
//     appointmentsCollection
//       .find(filter)
//       .sort({
//         createdAt: -1,
//         _id: -1,
//       })
//       .skip((page - 1) * limit)
//       .limit(limit)
//       .toArray(),

//     appointmentsCollection.countDocuments(filter),
//   ]);

//   const appointmentsWithImages =
//     await attachPatientImages(appointmentDocuments);

//   res.status(200).json({
//     success: true,
//     appointments: appointmentsWithImages.map(formatAppointment),
//     pagination: {
//       page,
//       limit,
//       total,
//       totalPages: Math.max(1, Math.ceil(total / limit)),
//     },
//   });
// };

// /* =========================================================
//    Admin appointment management
// ========================================================= */

// app.get(
//   "/api/v1/admin/appointments",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       await sendAppointmentList(req, res);
//     } catch (error) {
//       console.error("Get admin appointments error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Doctor appointment management
// ========================================================= */

// app.get(
//   "/api/v1/doctor/appointments",
//   verifyToken,
//   verifyDoctor,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       await sendAppointmentList(req, res, {
//         doctorUserId,
//       });
//     } catch (error) {
//       console.error("Get doctor appointments error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointments",
//       });
//     }
//   },
// );

// /* =========================================================
//    Doctor single appointment details
// ========================================================= */

// app.get(
//   "/api/v1/doctor/appointments/:appointmentId",
//   verifyToken,
//   verifyDoctor,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       if (!appointmentId) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment ID is required",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       const appointment = await database.collection("appointments").findOne({
//         $and: [
//           getDoctorFilter(appointmentId),
//           {
//             doctorUserId,
//           },
//         ],
//       });

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const [appointmentWithImage] = await attachPatientImages([appointment]);

//       res.status(200).json({
//         success: true,
//         appointment: formatAppointment(appointmentWithImage),
//       });
//     } catch (error) {
//       console.error("Get doctor appointment details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve appointment details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// /* =========================================================
//    Doctor appointment reschedule
// ========================================================= */

// app.patch(
//   "/api/v1/doctor/appointments/:appointmentId/reschedule",
//   verifyToken,
//   verifyDoctor,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentDate = getDoctorString(req.body.appointmentDate);

//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       const rescheduleReason = getDoctorString(req.body.rescheduleReason);

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       if (rescheduleReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Reschedule reason cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const doctorUserId = getDoctorDocumentId(currentUser);

//       if (getDoctorString(appointment.doctorUserId) !== doctorUserId) {
//         res.status(403).json({
//           success: false,
//           message: "You can reschedule only your own appointments",
//         });

//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (
//         currentStatus !== "pending" &&
//         currentStatus !== "approved" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message:
//             "Only pending, approved or rejected appointments can be rescheduled",
//         });

//         return;
//       }

//       const now = new Date();

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: {
//             appointmentDate,
//             appointmentTime,
//             rescheduleReason: rescheduleReason || null,
//             rescheduledAt: now,
//             rescheduledBy: doctorUserId,
//             updatedAt: now,
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message: "Appointment rescheduled successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Doctor reschedule appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to reschedule appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin-only appointment reschedule
// ========================================================= */

// app.patch(
//   "/api/v1/admin/appointments/:appointmentId/reschedule",
//   verifyToken,
//   verifyAdmin,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentDate = getDoctorString(req.body.appointmentDate);

//       const appointmentTime = getDoctorString(req.body.appointmentTime);

//       const rescheduleReason = getDoctorString(req.body.rescheduleReason);

//       const datePattern = /^\d{4}-\d{2}-\d{2}$/;
//       const timePattern = /^\d{2}:\d{2}$/;

//       if (
//         !datePattern.test(appointmentDate) ||
//         !timePattern.test(appointmentTime)
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "A valid appointment date and time are required",
//         });

//         return;
//       }

//       const today = new Date().toISOString().slice(0, 10);

//       if (appointmentDate < today) {
//         res.status(400).json({
//           success: false,
//           message: "Appointment date cannot be in the past",
//         });

//         return;
//       }

//       if (rescheduleReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Reschedule reason cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed" || currentStatus === "rejected") {
//         res.status(409).json({
//           success: false,
//           message: "A completed or rejected appointment cannot be rescheduled",
//         });

//         return;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: {
//             appointmentDate,
//             appointmentTime,
//             rescheduleReason: rescheduleReason || null,
//             rescheduledAt: new Date(),
//             rescheduledBy: req.userId || null,
//             updatedAt: new Date(),
//           },
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message: "Appointment rescheduled successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Reschedule appointment error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to reschedule appointment",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// app.patch(
//   "/api/v1/appointments/:appointmentId/status",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });

//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const userStatus = getNormalizedUserStatus(currentUser);

//       if (role !== "admin" && role !== "doctor") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Only an administrator or doctor can update appointment status.",
//         });

//         return;
//       }

//       if (userStatus === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Your account is blocked. You can view appointments but cannot update them.",
//           code: "READ_ONLY_ACCOUNT",
//         });

//         return;
//       }

//       const requestedStatus = getDoctorString(req.body.status);
//       const rejectionReason = getDoctorString(req.body.rejectionReason);

//       if (
//         requestedStatus !== "approved" &&
//         requestedStatus !== "completed" &&
//         requestedStatus !== "rejected"
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "Status must be approved, completed or rejected",
//         });

//         return;
//       }

//       if (requestedStatus === "rejected" && !rejectionReason) {
//         res.status(400).json({
//           success: false,
//           message:
//             "A rejection message is required when rejecting an appointment",
//         });

//         return;
//       }

//       if (rejectionReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Rejection message cannot contain more than 1000 characters",
//         });

//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);

//       const appointmentsCollection = database.collection("appointments");

//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });

//         return;
//       }

//       if (role === "doctor") {
//         const currentUserId = getDoctorDocumentId(currentUser);

//         if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
//           res.status(403).json({
//             success: false,
//             message: "You can update only your own appointments",
//           });

//           return;
//         }
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be changed",
//         });

//         return;
//       }

//       if (requestedStatus === "completed" && currentStatus !== "approved") {
//         res.status(409).json({
//           success: false,
//           message: "Only an approved appointment can be marked as completed",
//         });

//         return;
//       }

//       if (
//         requestedStatus === "approved" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or rejected appointment can be approved",
//         });

//         return;
//       }

//       if (
//         requestedStatus === "rejected" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "approved"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or approved appointment can be rejected",
//         });

//         return;
//       }

//       if (requestedStatus === "approved" && currentStatus === "rejected") {
//         const anotherActiveAppointment = await appointmentsCollection.findOne({
//           _id: {
//             $ne: appointment._id,
//           },
//           doctorId: getDoctorString(appointment.doctorId),
//           patientUserId: getDoctorString(appointment.patientUserId),
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "This patient already has another pending or approved appointment with you.",
//           });

//           return;
//         }
//       }

//       const now = new Date();

//       const statusFields: Record<string, unknown> = {
//         status: requestedStatus as AppointmentStatus,
//         rejectionReason:
//           requestedStatus === "rejected" ? rejectionReason : null,
//         updatedAt: now,
//       };

//       if (requestedStatus === "approved") {
//         statusFields.approvedAt = now;
//         statusFields.rejectedAt = null;
//         statusFields.rejectionReason = null;
//       }

//       if (requestedStatus === "completed") {
//         statusFields.completedAt = now;
//       }

//       if (requestedStatus === "rejected") {
//         statusFields.rejectedAt = now;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: statusFields,
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message:
//           requestedStatus === "approved"
//             ? "Appointment approved successfully"
//             : requestedStatus === "completed"
//               ? "Consultation completed successfully."
//               : "Appointment rejected successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update appointment status error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment status",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin or doctor appointment status update
// ========================================================= */

// app.patch(
//   "/api/v1/appointments/:appointmentId/status",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const userStatus = getNormalizedUserStatus(currentUser);

//       if (role !== "admin" && role !== "doctor") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Only an administrator or doctor can update appointment status.",
//         });
//         return;
//       }

//       if (userStatus === "blocked") {
//         res.status(403).json({
//           success: false,
//           message:
//             "Your account is blocked. You can view appointments but cannot update them.",
//           code: "READ_ONLY_ACCOUNT",
//         });
//         return;
//       }

//       const requestedStatus = getDoctorString(req.body.status);
//       const rejectionReason = getDoctorString(req.body.rejectionReason);

//       if (
//         requestedStatus !== "approved" &&
//         requestedStatus !== "completed" &&
//         requestedStatus !== "rejected"
//       ) {
//         res.status(400).json({
//           success: false,
//           message: "Status must be approved, completed or rejected",
//         });
//         return;
//       }

//       if (requestedStatus === "rejected" && !rejectionReason) {
//         res.status(400).json({
//           success: false,
//           message:
//             "A rejection message is required when rejecting an appointment",
//         });
//         return;
//       }

//       if (rejectionReason.length > 1000) {
//         res.status(400).json({
//           success: false,
//           message: "Rejection message cannot contain more than 1000 characters",
//         });
//         return;
//       }

//       const appointmentId = getDoctorString(req.params.appointmentId);
//       const appointmentsCollection = database.collection("appointments");
//       const appointment = await appointmentsCollection.findOne(
//         getDoctorFilter(appointmentId),
//       );

//       if (!appointment) {
//         res.status(404).json({
//           success: false,
//           message: "Appointment was not found",
//         });
//         return;
//       }

//       if (role === "doctor") {
//         const currentUserId = getDoctorDocumentId(currentUser);
//         if (getDoctorString(appointment.doctorUserId) !== currentUserId) {
//           res.status(403).json({
//             success: false,
//             message: "You can update only your own appointments",
//           });
//           return;
//         }
//       }

//       const currentStatus = getDoctorString(appointment.status);

//       if (currentStatus === "completed") {
//         res.status(409).json({
//           success: false,
//           message: "A completed appointment cannot be changed",
//         });
//         return;
//       }

//       if (requestedStatus === "completed" && currentStatus !== "approved") {
//         res.status(409).json({
//           success: false,
//           message: "Only an approved appointment can be marked as completed",
//         });
//         return;
//       }

//       if (
//         requestedStatus === "approved" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "rejected"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or rejected appointment can be approved",
//         });
//         return;
//       }

//       if (
//         requestedStatus === "rejected" &&
//         currentStatus !== "pending" &&
//         currentStatus !== "approved"
//       ) {
//         res.status(409).json({
//           success: false,
//           message: "Only a pending or approved appointment can be rejected",
//         });
//         return;
//       }

//       if (requestedStatus === "approved" && currentStatus === "rejected") {
//         const anotherActiveAppointment = await appointmentsCollection.findOne({
//           _id: {
//             $ne: appointment._id,
//           },
//           doctorId: getDoctorString(appointment.doctorId),
//           patientUserId: getDoctorString(appointment.patientUserId),
//           status: {
//             $in: ACTIVE_APPOINTMENT_STATUSES,
//           },
//         });

//         if (anotherActiveAppointment) {
//           res.status(409).json({
//             success: false,
//             message:
//               "This patient already has another pending or approved appointment with you.",
//           });
//           return;
//         }
//       }

//       const now = new Date();

//       const statusFields: Record<string, unknown> = {
//         status: requestedStatus as AppointmentStatus,
//         rejectionReason:
//           requestedStatus === "rejected" ? rejectionReason : null,
//         updatedAt: now,
//       };

//       if (requestedStatus === "approved") {
//         statusFields.approvedAt = now;
//         statusFields.rejectedAt = null;
//         statusFields.rejectionReason = null;
//       }

//       if (requestedStatus === "completed") {
//         statusFields.completedAt = now;
//       }

//       if (requestedStatus === "rejected") {
//         statusFields.rejectedAt = now;
//       }

//       const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
//         {
//           _id: appointment._id,
//         },
//         {
//           $set: statusFields,
//         },
//         {
//           returnDocument: "after",
//         },
//       );

//       res.status(200).json({
//         success: true,
//         message:
//           requestedStatus === "approved"
//             ? "Appointment approved successfully"
//             : requestedStatus === "completed"
//               ? "Consultation completed successfully."
//               : "Appointment rejected successfully",
//         appointment: updatedAppointment
//           ? formatAppointment(updatedAppointment)
//           : null,
//       });
//     } catch (error) {
//       console.error("Update appointment status error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to update appointment status",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin Dashboard Statistics                               <-- ADD THIS HERE
// ========================================================= */

// app.get(
//   "/api/v1/admin/dashboard/stats",
//   verifyToken,
//   verifyAdmin,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     console.log("✅ Admin dashboard stats route called!"); // Debug log
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       // Get total patients
//       const totalPatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient" });

//       // Get active patients
//       const activePatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient", status: "active" });

//       // Get blocked patients
//       const blockedPatients = await database
//         .collection("user")
//         .countDocuments({ role: "patient", status: "blocked" });

//       // Get total doctors
//       const totalDoctors = await database
//         .collection("doctors")
//         .countDocuments();

//       // Get active doctors
//       const activeDoctors = await database
//         .collection("doctors")
//         .countDocuments({ status: "active" });

//       // Get blocked doctors
//       const blockedDoctors = await database
//         .collection("doctors")
//         .countDocuments({ status: "blocked" });

//       // Get appointment counts by status
//       const appointmentCounts = await database
//         .collection("appointments")
//         .aggregate([
//           {
//             $group: {
//               _id: "$status",
//               count: { $sum: 1 },
//             },
//           },
//         ])
//         .toArray();

//       // Create status count object with default values
//       const statusCounts: Record<string, number> = {
//         pending: 0,
//         approved: 0,
//         completed: 0,
//         rejected: 0,
//       };

//       appointmentCounts.forEach((item) => {
//         const status = item._id || "pending";
//         if (status in statusCounts) {
//           statusCounts[status] = item.count;
//         }
//       });

//       // Get total appointments
//       const totalAppointments = await database
//         .collection("appointments")
//         .countDocuments();

//       // Get completed consultations
//       const completedConsultations = statusCounts.completed;

//       // Get monthly appointment trends (last 6 months)
//       const sixMonthsAgo = new Date();
//       sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

//       const monthlyTrends = await database
//         .collection("appointments")
//         .aggregate([
//           {
//             $match: {
//               createdAt: { $gte: sixMonthsAgo },
//             },
//           },
//           {
//             $group: {
//               _id: {
//                 year: { $year: "$createdAt" },
//                 month: { $month: "$createdAt" },
//               },
//               count: { $sum: 1 },
//               pending: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
//                 },
//               },
//               approved: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
//                 },
//               },
//               completed: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
//                 },
//               },
//               rejected: {
//                 $sum: {
//                   $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
//                 },
//               },
//             },
//           },
//           {
//             $sort: { "_id.year": 1, "_id.month": 1 },
//           },
//         ])
//         .toArray();

//       // Get appointment status breakdown for charts
//       const statusColors: Record<string, string> = {
//         pending: "#FBBF24",
//         approved: "#60A5FA",
//         completed: "#34D399",
//         rejected: "#F87171",
//       };

//       const statusData = appointmentCounts.map((item) => ({
//         name: item._id || "unknown",
//         value: item.count,
//         fill: statusColors[item._id] || "#9CA3AF",
//       }));

//       // Format monthly data for charts
//       const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
//       const monthlyData = monthlyTrends.map((item) => ({
//         month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
//         pending: item.pending || 0,
//         approved: item.approved || 0,
//         completed: item.completed || 0,
//         rejected: item.rejected || 0,
//         total: item.count || 0,
//       }));

//       // Get recent appointments (last 10)
//       const recentAppointments = await database
//         .collection("appointments")
//         .find()
//         .sort({ createdAt: -1 })
//         .limit(10)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         data: {
//           overview: {
//             totalPatients,
//             activePatients,
//             blockedPatients,
//             totalDoctors,
//             activeDoctors,
//             blockedDoctors,
//             totalAppointments,
//             completedConsultations,
//             appointmentStatus: statusCounts,
//           },
//           charts: {
//             appointmentStatus: statusData,
//             monthlyTrends: monthlyData,
//           },
//           recentAppointments: recentAppointments.map((app) => ({
//             id: app._id,
//             patientName: app.patientName,
//             doctorName: app.doctorName,
//             specialization: app.specialization,
//             appointmentDate: app.appointmentDate,
//             appointmentTime: app.appointmentTime,
//             status: app.status,
//             createdAt: app.createdAt,
//           })),
//         },
//       });
//     } catch (error) {
//       console.error("Dashboard stats error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to fetch dashboard statistics",
//       });
//     }
//   },
// );

// /* =========================================================
//    Admin Dashboard Statistics                         <-- ADD THIS HERE
// ========================================================= */

// // app.get(
// //   "/api/v1/admin/dashboard/stats",
// //   verifyToken,
// //   verifyAdmin,
// //   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
// //     console.log("✅ Admin dashboard stats route called!"); // Debug log
// //     try {
// //       if (!database) {
// //         res.status(503).json({
// //           success: false,
// //           message: "Database is not connected",
// //         });
// //         return;
// //       }

// //       // Get total patients
// //       const totalPatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient" });

// //       // Get active patients
// //       const activePatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient", status: "active" });

// //       // Get blocked patients
// //       const blockedPatients = await database
// //         .collection("user")
// //         .countDocuments({ role: "patient", status: "blocked" });

// //       // Get total doctors
// //       const totalDoctors = await database
// //         .collection("doctors")
// //         .countDocuments();

// //       // Get active doctors
// //       const activeDoctors = await database
// //         .collection("doctors")
// //         .countDocuments({ status: "active" });

// //       // Get blocked doctors
// //       const blockedDoctors = await database
// //         .collection("doctors")
// //         .countDocuments({ status: "blocked" });

// //       // Get appointment counts by status
// //       const appointmentCounts = await database
// //         .collection("appointments")
// //         .aggregate([
// //           {
// //             $group: {
// //               _id: "$status",
// //               count: { $sum: 1 },
// //             },
// //           },
// //         ])
// //         .toArray();

// //       // Create status count object with default values
// //       const statusCounts: Record<string, number> = {
// //         pending: 0,
// //         approved: 0,
// //         completed: 0,
// //         rejected: 0,
// //       };

// //       appointmentCounts.forEach((item) => {
// //         const status = item._id || "pending";
// //         if (status in statusCounts) {
// //           statusCounts[status] = item.count;
// //         }
// //       });

// //       // Get total appointments
// //       const totalAppointments = await database
// //         .collection("appointments")
// //         .countDocuments();

// //       // Get completed consultations
// //       const completedConsultations = statusCounts.completed;

// //       // Get monthly appointment trends (last 6 months)
// //       const sixMonthsAgo = new Date();
// //       sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);

// //       const monthlyTrends = await database
// //         .collection("appointments")
// //         .aggregate([
// //           {
// //             $match: {
// //               createdAt: { $gte: sixMonthsAgo },
// //             },
// //           },
// //           {
// //             $group: {
// //               _id: {
// //                 year: { $year: "$createdAt" },
// //                 month: { $month: "$createdAt" },
// //               },
// //               count: { $sum: 1 },
// //               pending: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "pending"] }, 1, 0],
// //                 },
// //               },
// //               approved: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "approved"] }, 1, 0],
// //                 },
// //               },
// //               completed: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "completed"] }, 1, 0],
// //                 },
// //               },
// //               rejected: {
// //                 $sum: {
// //                   $cond: [{ $eq: ["$status", "rejected"] }, 1, 0],
// //                 },
// //               },
// //             },
// //           },
// //           {
// //             $sort: { "_id.year": 1, "_id.month": 1 },
// //           },
// //         ])
// //         .toArray();

// //       // Get appointment status breakdown for charts
// //       const statusColors: Record<string, string> = {
// //         pending: "#FBBF24",
// //         approved: "#60A5FA",
// //         completed: "#34D399",
// //         rejected: "#F87171",
// //       };

// //       const statusData = appointmentCounts.map((item) => ({
// //         name: item._id || "unknown",
// //         value: item.count,
// //         fill: statusColors[item._id] || "#9CA3AF",
// //       }));

// //       // Format monthly data for charts
// //       const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// //       const monthlyData = monthlyTrends.map((item) => ({
// //         month: `${monthNames[item._id.month - 1]} ${item._id.year}`,
// //         pending: item.pending || 0,
// //         approved: item.approved || 0,
// //         completed: item.completed || 0,
// //         rejected: item.rejected || 0,
// //         total: item.count || 0,
// //       }));

// //       // Get recent appointments (last 10)
// //       const recentAppointments = await database
// //         .collection("appointments")
// //         .find()
// //         .sort({ createdAt: -1 })
// //         .limit(10)
// //         .toArray();

// //       res.status(200).json({
// //         success: true,
// //         data: {
// //           overview: {
// //             totalPatients,
// //             activePatients,
// //             blockedPatients,
// //             totalDoctors,
// //             activeDoctors,
// //             blockedDoctors,
// //             totalAppointments,
// //             completedConsultations,
// //             appointmentStatus: statusCounts,
// //           },
// //           charts: {
// //             appointmentStatus: statusData,
// //             monthlyTrends: monthlyData,
// //           },
// //           recentAppointments: recentAppointments.map((app) => ({
// //             id: app._id,
// //             patientName: app.patientName,
// //             doctorName: app.doctorName,
// //             specialization: app.specialization,
// //             appointmentDate: app.appointmentDate,
// //             appointmentTime: app.appointmentTime,
// //             status: app.status,
// //             createdAt: app.createdAt,
// //           })),
// //         },
// //       });
// //     } catch (error) {
// //       console.error("Dashboard stats error:", error);
// //       res.status(500).json({
// //         success: false,
// //         message: "Failed to fetch dashboard statistics",
// //       });
// //     }
// //   },
// // );

// /* =========================================================
//    SebaSathi AI Health Assistant (Groq)
// ========================================================= */

// type AIHealthMessageRole = "user" | "assistant";

// type AIHealthUrgency = "routine" | "soon" | "urgent" | "emergency";

// type AIHealthStreamStage =
//   | "thinking"
//   | "tool"
//   | "answering"
//   | "structuring"
//   | "saving";

// interface AIHealthMessage {
//   role: AIHealthMessageRole;
//   content: string;
// }

// interface AIHealthNavigationRoute {
//   label: string;
//   href: string;
//   description: string;
// }

// interface AIHealthNavigationAction {
//   label: string;
//   href: string;
//   reason: string;
// }

// interface AIHealthAssistantResponse {
//   reply: string;
//   urgencyLevel: AIHealthUrgency;
//   suggestedSpecialists: string[];
//   recommendedActions: string[];
//   warningSigns: string[];
//   followUpQuestions: string[];
//   suggestedPrompts: string[];
//   navigationActions: AIHealthNavigationAction[];
//   decisionBasis: string;
//   toolsUsed: string[];
//   contextMemoryUsed: boolean;
//   disclaimer: string;
// }

// interface AIHealthStoredMessage extends AIHealthMessage {
//   id: string;
//   assistant?: AIHealthAssistantResponse;
//   createdAt: Date;
// }

// interface AIHealthSummaryReport {
//   reportTitle: string;
//   conciseSummary: string;
//   chiefConcerns: string[];
//   symptoms: string[];
//   durationAndPattern: string;
//   severity: string;
//   urgencyLevel: AIHealthUrgency;
//   redFlags: string[];
//   suggestedSpecialists: string[];
//   selfCareGuidance: string[];
//   questionsForDoctor: string[];
//   emergencyAdvice: string;
//   disclaimer: string;
// }

// interface AIHealthApplicationContext {
//   user: {
//     id: string;
//     name: string;
//     role: UserRole;
//   };
//   routes: AIHealthNavigationRoute[];
//   doctorDirectory: {
//     activeDoctorCount: number;
//     specializations: string[];
//     highlightedDoctors: Array<{
//       id: string;
//       name: string;
//       specialization: string;
//       hospital: string;
//       ratingAverage: number;
//     }>;
//   } | null;
//   appointmentContext: {
//     total: number;
//     counts: Record<string, number>;
//     recentAppointments: Array<{
//       id: string;
//       doctorName: string;
//       patientName: string;
//       specialization: string;
//       appointmentDate: string;
//       appointmentTime: string;
//       status: string;
//     }>;
//   } | null;
//   recentHealthHistory: Array<{
//     id: string;
//     title: string;
//     urgencyLevel: AIHealthUrgency;
//     updatedAt: string | null;
//   }>;
//   toolsUsed: string[];
//   contextMemoryUsed: boolean;
// }

// const aiHealthRateLimit = new Map<
//   string,
//   {
//     startedAt: number;
//     count: number;
//   }
// >();

// const verifyAIHealthRateLimit = (
//   req: AuthenticatedRequest,
//   res: Response,
//   next: NextFunction,
// ): void => {
//   const key = req.userId || req.ip || "anonymous";
//   const now = Date.now();
//   const windowLength = 10 * 60 * 1000;
//   const maximumRequests = 40;
//   const current = aiHealthRateLimit.get(key);

//   if (!current || now - current.startedAt >= windowLength) {
//     aiHealthRateLimit.set(key, {
//       startedAt: now,
//       count: 1,
//     });

//     next();
//     return;
//   }

//   if (current.count >= maximumRequests) {
//     res.status(429).json({
//       success: false,
//       message:
//         "You have sent too many AI requests. Please try again after a few minutes.",
//       code: "AI_RATE_LIMITED",
//     });

//     return;
//   }

//   current.count += 1;
//   aiHealthRateLimit.set(key, current);
//   next();
// };

// const createAIHealthMessageId = (): string => {
//   return new ObjectId().toHexString();
// };

// const getAIHealthArray = (value: unknown, maximumItems = 8): string[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   return value
//     .map((item) => getDoctorString(item))
//     .filter(Boolean)
//     .slice(0, maximumItems);
// };

// const getAIHealthUrgency = (value: unknown): AIHealthUrgency => {
//   return value === "soon" || value === "urgent" || value === "emergency"
//     ? value
//     : "routine";
// };

// const getAIHealthBoolean = (value: unknown): boolean => value === true;

// const extractAIHealthJson = (content: string): Record<string, unknown> => {
//   const trimmed = content.trim();
//   const withoutFence = trimmed
//     .replace(/^```(?:json)?\s*/i, "")
//     .replace(/\s*```$/i, "")
//     .trim();

//   try {
//     const parsed = JSON.parse(withoutFence) as unknown;

//     if (typeof parsed === "object" && parsed !== null) {
//       return parsed as Record<string, unknown>;
//     }
//   } catch {
//     const firstBrace = withoutFence.indexOf("{");
//     const lastBrace = withoutFence.lastIndexOf("}");

//     if (firstBrace >= 0 && lastBrace > firstBrace) {
//       const parsed = JSON.parse(
//         withoutFence.slice(firstBrace, lastBrace + 1),
//       ) as unknown;

//       if (typeof parsed === "object" && parsed !== null) {
//         return parsed as Record<string, unknown>;
//       }
//     }
//   }

//   throw new Error("Groq returned an invalid structured response");
// };

// const normalizeAIHealthMessages = (
//   value: unknown,
//   options: {
//     requireLatestUser: boolean;
//     maximumMessages?: number;
//     maximumCharacters?: number;
//   },
// ):
//   | {
//       success: true;
//       messages: AIHealthMessage[];
//     }
//   | {
//       success: false;
//       message: string;
//     } => {
//   if (!Array.isArray(value)) {
//     return {
//       success: false,
//       message: "A conversation message list is required",
//     };
//   }

//   const maximumMessages = options.maximumMessages ?? 30;
//   const maximumCharacters = options.maximumCharacters ?? 30000;
//   const messages: AIHealthMessage[] = [];
//   let totalCharacters = 0;

//   for (const rawMessage of value.slice(-maximumMessages)) {
//     if (typeof rawMessage !== "object" || rawMessage === null) {
//       continue;
//     }

//     const message = rawMessage as Record<string, unknown>;
//     const role = message.role;
//     const content = getDoctorString(message.content);

//     if ((role !== "user" && role !== "assistant") || !content) {
//       continue;
//     }

//     if (content.length > 4000) {
//       return {
//         success: false,
//         message: "Each chat message cannot contain more than 4000 characters",
//       };
//     }

//     totalCharacters += content.length;

//     if (totalCharacters > maximumCharacters) {
//       return {
//         success: false,
//         message:
//           "This conversation is too long. Please generate a summary and start a new conversation.",
//       };
//     }

//     messages.push({
//       role,
//       content,
//     });
//   }

//   if (messages.length === 0) {
//     return {
//       success: false,
//       message: "At least one valid chat message is required",
//     };
//   }

//   if (!messages.some((message) => message.role === "user")) {
//     return {
//       success: false,
//       message: "At least one user message is required",
//     };
//   }

//   if (
//     options.requireLatestUser &&
//     messages[messages.length - 1]?.role !== "user"
//   ) {
//     return {
//       success: false,
//       message: "The latest conversation message must be from the user",
//     };
//   }

//   return {
//     success: true,
//     messages,
//   };
// };

// const PUBLIC_AI_HEALTH_NAVIGATION_ROUTES: AIHealthNavigationRoute[] = [
//   {
//     label: "Home",
//     href: "/",
//     description: "Open the SebaSathi home page.",
//   },
//   {
//     label: "Find Doctors",
//     href: "/find-doctors",
//     description: "Find active doctors and filter by specialization.",
//   },
//   {
//     label: "AI Health Assistant",
//     href: "/ai-health-assistant",
//     description: "Continue using the SebaSathi AI Health Assistant.",
//   },
//   {
//     label: "About Us",
//     href: "/about",
//     description: "Learn more about SebaSathi and its healthcare services.",
//   },
//   {
//     label: "Contact",
//     href: "/contact",
//     description: "Open the SebaSathi contact page.",
//   },
// ];

// const ROLE_AI_HEALTH_NAVIGATION_ROUTES: Record<
//   UserRole,
//   AIHealthNavigationRoute[]
// > = {
//   patient: [
//     {
//       label: "Patient Overview",
//       href: "/dashboard/patient",
//       description: "Open the patient's dashboard overview.",
//     },
//     {
//       label: "My Appointments",
//       href: "/dashboard/patient/appointments",
//       description: "View the patient's appointment requests and statuses.",
//     },
//     {
//       label: "Prescriptions",
//       href: "/dashboard/patient/prescriptions",
//       description: "View the patient's saved prescriptions.",
//     },
//     {
//       label: "Consultations",
//       href: "/dashboard/patient/consultations",
//       description: "View the patient's consultation records.",
//     },
//     {
//       label: "AI Health History",
//       href: "/dashboard/patient/ai-health-history",
//       description: "Review saved AI-generated health summaries.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/patient/my-profile",
//       description: "Open the patient's profile settings.",
//     },
//   ],
//   doctor: [
//     {
//       label: "Doctor Overview",
//       href: "/dashboard/doctor",
//       description: "Open the doctor's dashboard overview.",
//     },
//     {
//       label: "Appointments",
//       href: "/dashboard/doctor/patients-appointments",
//       description: "View appointments assigned to the signed-in doctor.",
//     },
//     {
//       label: "My Patients",
//       href: "/dashboard/doctor/patients",
//       description: "View the doctor's patient list.",
//     },
//     {
//       label: "Prescriptions",
//       href: "/dashboard/doctor/prescriptions",
//       description: "Create or review doctor prescription records.",
//     },
//     {
//       label: "Consultation Records",
//       href: "/dashboard/doctor/consultations",
//       description: "View the doctor's consultation records.",
//     },
//     {
//       label: "Availability",
//       href: "/dashboard/doctor/availability",
//       description: "Manage the doctor's availability schedule.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/doctor/my-profile",
//       description: "Open the doctor's profile settings.",
//     },
//   ],
//   admin: [
//     {
//       label: "Admin Overview",
//       href: "/dashboard/admin",
//       description: "Open the administrator dashboard overview.",
//     },
//     {
//       label: "Manage Users",
//       href: "/dashboard/admin/users",
//       description: "Open administrator user management.",
//     },
//     {
//       label: "Manage Doctors",
//       href: "/dashboard/admin/doctors",
//       description: "Open administrator doctor management.",
//     },
//     {
//       label: "Manage Appointments",
//       href: "/dashboard/admin/appointments",
//       description: "Open administrator appointment management.",
//     },
//     {
//       label: "My Profile",
//       href: "/dashboard/admin/my-profile",
//       description: "Open the administrator's profile settings.",
//     },
//   ],
// };

// const AI_HEALTH_NAVIGATION_ROUTE_ALIASES: Record<string, string> = {
//   "/doctors": "/find-doctors",
//   "/dashboard/doctor/appointments": "/dashboard/doctor/patients-appointments",
// };

// const normalizeAIHealthNavigationHref = (href: string): string => {
//   return AI_HEALTH_NAVIGATION_ROUTE_ALIASES[href] || href;
// };

// const getAllAIHealthNavigationRoutes = (): AIHealthNavigationRoute[] => [
//   ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.patient,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.doctor,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES.admin,
// ];

// const getAIHealthNavigationRoutes = (
//   role: UserRole,
// ): AIHealthNavigationRoute[] => [
//   ...PUBLIC_AI_HEALTH_NAVIGATION_ROUTES,
//   ...ROLE_AI_HEALTH_NAVIGATION_ROUTES[role],
// ];

// const getAIHealthNavigationActions = (
//   value: unknown,
//   allowedRoutes: AIHealthNavigationRoute[],
// ): AIHealthNavigationAction[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   const allowedByHref = new Map(
//     allowedRoutes.map((route) => [route.href, route] as const),
//   );

//   const actions: AIHealthNavigationAction[] = [];

//   for (const rawAction of value) {
//     if (typeof rawAction !== "object" || rawAction === null) {
//       continue;
//     }

//     const action = rawAction as Record<string, unknown>;
//     const href = normalizeAIHealthNavigationHref(getDoctorString(action.href));
//     const allowedRoute = allowedByHref.get(href);

//     if (!allowedRoute) {
//       continue;
//     }

//     actions.push({
//       label: getDoctorString(action.label) || allowedRoute.label,
//       href,
//       reason: getDoctorString(action.reason) || allowedRoute.description,
//     });

//     if (actions.length >= 3) {
//       break;
//     }
//   }

//   return actions;
// };

// const formatAIHealthAssistantResponse = (
//   data: Record<string, unknown>,
//   emergencyDetected: boolean,
//   context?: AIHealthApplicationContext,
// ): AIHealthAssistantResponse => {
//   const urgencyLevel = emergencyDetected
//     ? "emergency"
//     : getAIHealthUrgency(data.urgencyLevel);

//   const reply =
//     getDoctorString(data.reply) ||
//     "Please describe the symptoms, duration and severity a little more clearly.";

//   const followUpQuestions = getAIHealthArray(data.followUpQuestions, 3);
//   const suggestedPrompts = getAIHealthArray(data.suggestedPrompts, 4);
//   const allowedRoutes = context?.routes || getAllAIHealthNavigationRoutes();
//   const toolsUsed = context?.toolsUsed.length
//     ? context.toolsUsed
//     : getAIHealthArray(data.toolsUsed, 8);
//   const contextMemoryUsed = context
//     ? context.contextMemoryUsed
//     : getAIHealthBoolean(data.contextMemoryUsed);

//   return {
//     reply,
//     urgencyLevel,
//     suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 3),
//     recommendedActions: getAIHealthArray(data.recommendedActions, 5),
//     warningSigns: emergencyDetected
//       ? Array.from(
//           new Set([
//             "Your description may include an emergency warning sign.",
//             ...getAIHealthArray(data.warningSigns, 4),
//           ]),
//         ).slice(0, 4)
//       : getAIHealthArray(data.warningSigns, 4),
//     followUpQuestions,
//     suggestedPrompts:
//       suggestedPrompts.length > 0
//         ? suggestedPrompts
//         : followUpQuestions.slice(0, 3),
//     navigationActions: getAIHealthNavigationActions(
//       data.navigationActions,
//       allowedRoutes,
//     ),
//     decisionBasis:
//       getDoctorString(data.decisionBasis) ||
//       "This guidance is based on the symptoms, duration, severity, warning signs and relevant SebaSathi application context available in this conversation.",
//     toolsUsed,
//     contextMemoryUsed,
//     disclaimer:
//       getDoctorString(data.disclaimer) ||
//       "General guidance only; this is not a diagnosis or prescription.",
//   };
// };

// const getStoredAIHealthMessages = (value: unknown): AIHealthStoredMessage[] => {
//   if (!Array.isArray(value)) {
//     return [];
//   }

//   return value
//     .map((rawMessage): AIHealthStoredMessage | null => {
//       if (typeof rawMessage !== "object" || rawMessage === null) {
//         return null;
//       }

//       const message = rawMessage as Record<string, unknown>;
//       const role = message.role;
//       const content = getDoctorString(message.content);

//       if ((role !== "user" && role !== "assistant") || !content) {
//         return null;
//       }

//       const assistant =
//         typeof message.assistant === "object" && message.assistant !== null
//           ? formatAIHealthAssistantResponse(
//               message.assistant as Record<string, unknown>,
//               false,
//             )
//           : undefined;

//       const createdAtValue = message.createdAt;
//       const createdAt =
//         createdAtValue instanceof Date
//           ? createdAtValue
//           : new Date(
//               typeof createdAtValue === "string" ||
//                 typeof createdAtValue === "number"
//                 ? createdAtValue
//                 : Date.now(),
//             );

//       return {
//         id: getDoctorString(message.id) || createAIHealthMessageId(),
//         role,
//         content,
//         ...(assistant ? { assistant } : {}),
//         createdAt: Number.isNaN(createdAt.getTime()) ? new Date() : createdAt,
//       };
//     })
//     .filter((message): message is AIHealthStoredMessage => message !== null);
// };

// const hasEmergencyWarning = (messages: AIHealthMessage[]): boolean => {
//   const text = messages
//     .filter((message) => message.role === "user")
//     .map((message) => message.content)
//     .join(" ")
//     .toLowerCase();

//   const emergencyPatterns = [
//     /severe chest pain/,
//     /cannot breathe/,
//     /can't breathe/,
//     /difficulty breathing/,
//     /heavy bleeding/,
//     /unconscious/,
//     /not responding/,
//     /seizure/,
//     /stroke symptoms/,
//     /face droop/,
//     /suicid(?:e|al)/,
//     /kill myself/,
//     /বুকে তীব্র ব্যথা/,
//     /শ্বাস নিতে পারছি না/,
//     /শ্বাসকষ্ট/,
//     /অতিরিক্ত রক্তপাত/,
//     /অজ্ঞান/,
//     /খিঁচুনি/,
//     /আত্মহত্যা/,
//   ];

//   return emergencyPatterns.some((pattern) => pattern.test(text));
// };

// const callGroqAI = async (
//   messages: Array<{
//     role: "system" | "user" | "assistant";
//     content: string;
//   }>,
//   temperature: number,
//   maximumOutputTokens: number,
// ): Promise<Record<string, unknown>> => {
//   if (!groqApiKey) {
//     throw new Error("GROQ_API_KEY is missing from the backend .env file");
//   }

//   const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
//     method: "POST",
//     headers: {
//       authorization: `Bearer ${groqApiKey}`,
//       "content-type": "application/json",
//       accept: "application/json",
//     },
//     body: JSON.stringify({
//       model: groqModel,
//       messages,
//       temperature,
//       max_completion_tokens: maximumOutputTokens,
//       response_format: {
//         type: "json_object",
//       },
//     }),
//   });

//   const responseData = (await response.json().catch(() => null)) as Record<
//     string,
//     unknown
//   > | null;

//   if (!response.ok) {
//     const errorObject =
//       typeof responseData?.error === "object" && responseData.error !== null
//         ? (responseData.error as Record<string, unknown>)
//         : null;

//     const providerMessage = getDoctorString(errorObject?.message);

//     throw new Error(
//       providerMessage || `Groq request failed with status ${response.status}`,
//     );
//   }

//   const choices = Array.isArray(responseData?.choices)
//     ? responseData.choices
//     : [];

//   const firstChoice = choices[0];

//   if (typeof firstChoice !== "object" || firstChoice === null) {
//     throw new Error("Groq did not return an assistant response");
//   }

//   const choice = firstChoice as Record<string, unknown>;
//   const message =
//     typeof choice.message === "object" && choice.message !== null
//       ? (choice.message as Record<string, unknown>)
//       : null;

//   const content = getDoctorString(message?.content);

//   if (!content) {
//     throw new Error("Groq returned an empty assistant response");
//   }

//   return extractAIHealthJson(content);
// };

// const callGroqTextStream = async (
//   messages: Array<{
//     role: "system" | "user" | "assistant";
//     content: string;
//   }>,
//   onDelta: (delta: string) => void,
//   signal?: AbortSignal,
// ): Promise<string> => {
//   if (!groqApiKey) {
//     throw new Error("GROQ_API_KEY is missing from the backend .env file");
//   }

//   const response = await fetch(`${groqApiBaseUrl}/chat/completions`, {
//     method: "POST",
//     headers: {
//       authorization: `Bearer ${groqApiKey}`,
//       "content-type": "application/json",
//       accept: "text/event-stream",
//     },
//     body: JSON.stringify({
//       model: groqModel,
//       messages,
//       temperature: 0.25,
//       max_completion_tokens: 1100,
//       stream: true,
//     }),
//     signal,
//   });

//   if (!response.ok) {
//     const responseData = (await response.json().catch(() => null)) as Record<
//       string,
//       unknown
//     > | null;
//     const errorObject =
//       typeof responseData?.error === "object" && responseData.error !== null
//         ? (responseData.error as Record<string, unknown>)
//         : null;
//     const providerMessage = getDoctorString(errorObject?.message);

//     throw new Error(
//       providerMessage || `Groq request failed with status ${response.status}`,
//     );
//   }

//   if (!response.body) {
//     throw new Error("Groq streaming response body is unavailable");
//   }

//   const reader = response.body.getReader();
//   const decoder = new TextDecoder();
//   let buffer = "";
//   let completeText = "";

//   while (true) {
//     const { value, done } = await reader.read();

//     if (done) {
//       break;
//     }

//     buffer += decoder.decode(value, { stream: true });
//     const lines = buffer.split("\n");
//     buffer = lines.pop() || "";

//     for (const line of lines) {
//       const trimmed = line.trim();

//       if (!trimmed.startsWith("data:")) {
//         continue;
//       }

//       const payload = trimmed.slice(5).trim();

//       if (!payload || payload === "[DONE]") {
//         continue;
//       }

//       const parsed = JSON.parse(payload) as Record<string, unknown>;
//       const choices = Array.isArray(parsed.choices) ? parsed.choices : [];
//       const firstChoice = choices[0];

//       if (typeof firstChoice !== "object" || firstChoice === null) {
//         continue;
//       }

//       const delta = (firstChoice as Record<string, unknown>).delta;

//       if (typeof delta !== "object" || delta === null) {
//         continue;
//       }

//       const rawContent = (delta as Record<string, unknown>).content;
//       const content = typeof rawContent === "string" ? rawContent : "";

//       if (!content) {
//         continue;
//       }

//       completeText += content;
//       onDelta(content);
//     }
//   }

//   const finalText = completeText.trim();

//   if (!finalText) {
//     throw new Error("Groq returned an empty streamed response");
//   }

//   return finalText;
// };

// const formatAIHealthSummary = (
//   data: Record<string, unknown>,
// ): AIHealthSummaryReport => {
//   return {
//     reportTitle:
//       getDoctorString(data.reportTitle) || "AI Health Conversation Summary",
//     conciseSummary:
//       getDoctorString(data.conciseSummary) ||
//       "A concise summary could not be generated.",
//     chiefConcerns: getAIHealthArray(data.chiefConcerns, 6),
//     symptoms: getAIHealthArray(data.symptoms, 10),
//     durationAndPattern:
//       getDoctorString(data.durationAndPattern) || "Not clearly stated",
//     severity: getDoctorString(data.severity) || "Not clearly stated",
//     urgencyLevel: getAIHealthUrgency(data.urgencyLevel),
//     redFlags: getAIHealthArray(data.redFlags, 6),
//     suggestedSpecialists: getAIHealthArray(data.suggestedSpecialists, 5),
//     selfCareGuidance: getAIHealthArray(data.selfCareGuidance, 6),
//     questionsForDoctor: getAIHealthArray(data.questionsForDoctor, 6),
//     emergencyAdvice:
//       getDoctorString(data.emergencyAdvice) ||
//       "Seek urgent in-person medical care if symptoms become severe or new warning signs appear.",
//     disclaimer:
//       getDoctorString(data.disclaimer) ||
//       "This AI-generated summary is not a diagnosis or prescription.",
//   };
// };

// const createAIHealthConversationTitle = (message: string): string => {
//   const normalized = message.replace(/\s+/g, " ").trim();
//   const words = normalized.split(" ").filter(Boolean).slice(0, 7);
//   const title = words.join(" ");

//   if (!title) {
//     return "New health chat";
//   }

//   return normalized.length > title.length ? `${title}…` : title;
// };

// const getAIHealthOwnerFilter = (userId: string): Filter<Document> => {
//   return {
//     $or: [
//       {
//         userId,
//       },
//       {
//         patientUserId: userId,
//       },
//     ],
//   };
// };

// const formatAIHealthConversationMessage = (message: AIHealthStoredMessage) => {
//   return {
//     id: message.id,
//     role: message.role,
//     content: message.content,
//     assistant: message.assistant || null,
//     createdAt: formatDoctorDate(message.createdAt),
//   };
// };

// const formatAIHealthConversation = (conversation: Document) => {
//   const userId =
//     getDoctorString(conversation.userId) ||
//     getDoctorString(conversation.patientUserId);

//   const userRole: UserRole =
//     conversation.userRole === "admin" ||
//     conversation.userRole === "doctor" ||
//     conversation.userRole === "patient"
//       ? conversation.userRole
//       : "patient";

//   const messages = getStoredAIHealthMessages(conversation.messages);

//   return {
//     id: getDoctorDocumentId(conversation),
//     title: getDoctorString(conversation.title) || "New health chat",
//     userId,
//     userRole,
//     userName:
//       getDoctorString(conversation.userName) ||
//       getDoctorString(conversation.patientName),
//     userEmail:
//       normalizeDoctorEmail(conversation.userEmail) ||
//       normalizeDoctorEmail(conversation.patientEmail),
//     userImage:
//       getDoctorString(conversation.userImage) ||
//       getDoctorString(conversation.patientImage) ||
//       null,
//     messages: messages.map(formatAIHealthConversationMessage),
//     messageCount: messages.length,
//     summaryHistoryId: getDoctorString(conversation.summaryHistoryId) || null,
//     summaryReport:
//       typeof conversation.summaryReport === "object" &&
//       conversation.summaryReport !== null
//         ? conversation.summaryReport
//         : null,
//     createdAt: formatDoctorDate(conversation.createdAt),
//     updatedAt: formatDoctorDate(conversation.updatedAt),
//     lastMessageAt: formatDoctorDate(
//       conversation.lastMessageAt || conversation.updatedAt,
//     ),
//   };
// };

// const formatAIHealthHistory = (history: Document) => {
//   const userId =
//     getDoctorString(history.userId) || getDoctorString(history.patientUserId);

//   const userName =
//     getDoctorString(history.userName) || getDoctorString(history.patientName);

//   const userEmail =
//     normalizeDoctorEmail(history.userEmail) ||
//     normalizeDoctorEmail(history.patientEmail);

//   const userRole: UserRole =
//     history.userRole === "admin" ||
//     history.userRole === "doctor" ||
//     history.userRole === "patient"
//       ? history.userRole
//       : "patient";

//   return {
//     id: getDoctorDocumentId(history),
//     conversationId: getDoctorString(history.conversationId) || null,
//     conversationTitle: getDoctorString(history.conversationTitle) || null,
//     userId,
//     userRole,
//     userName,
//     userEmail,
//     userImage:
//       getDoctorString(history.userImage) ||
//       getDoctorString(history.patientImage) ||
//       null,
//     patientUserId: userId,
//     patientName: userName,
//     patientEmail: userEmail,
//     provider: getDoctorString(history.provider),
//     model: getDoctorString(history.model),
//     report:
//       typeof history.report === "object" && history.report !== null
//         ? history.report
//         : null,
//     messages: Array.isArray(history.messages) ? history.messages : [],
//     createdAt: formatDoctorDate(history.createdAt),
//     updatedAt: formatDoctorDate(history.updatedAt),
//   };
// };

// const getAIHealthConversationForUser = async (
//   userId: string,
//   conversationId: string,
// ): Promise<Document | null> => {
//   if (!database || !conversationId) {
//     return null;
//   }

//   return database.collection(AI_HEALTH_CHAT_COLLECTION).findOne({
//     $and: [getDoctorFilter(conversationId), getAIHealthOwnerFilter(userId)],
//   });
// };

// const detectAIHealthApplicationIntents = (message: string) => {
//   const normalized = message.toLowerCase();

//   return {
//     appointment:
//       /appointment|booking|schedule|pending|approved|rejected|অ্যাপয়েন্টমেন্ট|অ্যাপয়েন্টমেন্ট|বুকিং|সিডিউল|পেন্ডিং|এপ্রুভ/.test(
//         normalized,
//       ),
//     history:
//       /history|summary|report|previous chat|old chat|হিস্ট্রি|সামারি|রিপোর্ট|পুরোনো চ্যাট/.test(
//         normalized,
//       ),
//     navigation:
//       /open|go to|take me|navigate|where is|show page|dashboard|খুলে দাও|নিয়ে যাও|নিয়ে যাও|কোথায়|কোথায়|ড্যাশবোর্ড/.test(
//         normalized,
//       ),
//     doctor:
//       /doctor|specialist|specialization|cardio|derma|neuro|medicine|surgeon|ডাক্তার|বিশেষজ্ঞ|স্পেশালিস্ট|কার্ডিও|ডার্মা|নিউরো/.test(
//         normalized,
//       ),
//   };
// };

// const buildAIHealthApplicationContext = async (
//   currentUser: Document,
//   latestMessage: string,
//   existingMessages: AIHealthStoredMessage[],
// ): Promise<AIHealthApplicationContext> => {
//   if (!database) {
//     throw new Error("Database is not connected");
//   }

//   const userId = getDoctorDocumentId(currentUser);
//   const role = getNormalizedUserRole(currentUser);
//   const userName = getDoctorString(currentUser.name) || "User";
//   const intents = detectAIHealthApplicationIntents(latestMessage);
//   const routes = getAIHealthNavigationRoutes(role);
//   const toolsUsed = ["SebaSathi navigation map", "SebaSathi doctor directory"];
//   const contextMemoryUsed = existingMessages.length > 0;

//   if (contextMemoryUsed) {
//     toolsUsed.push("Conversation memory");
//   }

//   const [doctorDocuments, specializations, activeDoctorCount] =
//     await Promise.all([
//       database
//         .collection("doctors")
//         .find(
//           { status: "active" },
//           {
//             projection: {
//               name: 1,
//               specialization: 1,
//               hospital: 1,
//               chamber: 1,
//               ratingAverage: 1,
//             },
//           },
//         )
//         .sort({ ratingAverage: -1, ratingCount: -1, createdAt: -1 })
//         .limit(8)
//         .toArray(),
//       database.collection("doctors").distinct("specialization", {
//         status: "active",
//       }),
//       database.collection("doctors").countDocuments({ status: "active" }),
//     ]);

//   const doctorDirectory = {
//     activeDoctorCount,
//     specializations: specializations
//       .map((value) => getDoctorString(value))
//       .filter(Boolean)
//       .sort((a, b) => a.localeCompare(b))
//       .slice(0, 30),
//     highlightedDoctors: doctorDocuments.map((doctor) => ({
//       id: getDoctorDocumentId(doctor),
//       name: getDoctorString(doctor.name),
//       specialization: getDoctorString(doctor.specialization),
//       hospital:
//         getDoctorString(doctor.hospital) || getDoctorString(doctor.chamber),
//       ratingAverage: Number.isFinite(Number(doctor.ratingAverage))
//         ? Number(Number(doctor.ratingAverage).toFixed(1))
//         : 0,
//     })),
//   };

//   let appointmentContext: AIHealthApplicationContext["appointmentContext"] =
//     null;

//   if (intents.appointment || intents.navigation) {
//     toolsUsed.push("Appointment lookup");

//     const appointmentFilter: Filter<Document> =
//       role === "patient"
//         ? { patientUserId: userId }
//         : role === "doctor"
//           ? { doctorUserId: userId }
//           : {};

//     const [statusCounts, recentAppointments, total] = await Promise.all([
//       database
//         .collection("appointments")
//         .aggregate([
//           { $match: appointmentFilter },
//           {
//             $group: {
//               _id: "$status",
//               count: { $sum: 1 },
//             },
//           },
//         ])
//         .toArray(),
//       database
//         .collection("appointments")
//         .find(appointmentFilter)
//         .sort({ updatedAt: -1, createdAt: -1 })
//         .limit(5)
//         .toArray(),
//       database.collection("appointments").countDocuments(appointmentFilter),
//     ]);

//     appointmentContext = {
//       total,
//       counts: Object.fromEntries(
//         statusCounts.map((item) => [
//           getDoctorString(item._id) || "unknown",
//           Number(item.count) || 0,
//         ]),
//       ),
//       recentAppointments: recentAppointments.map((appointment) => ({
//         id: getDoctorDocumentId(appointment),
//         doctorName: getDoctorString(appointment.doctorName),
//         patientName: getDoctorString(appointment.patientName),
//         specialization: getDoctorString(appointment.specialization),
//         appointmentDate: getDoctorString(appointment.appointmentDate),
//         appointmentTime: getDoctorString(appointment.appointmentTime),
//         status: getDoctorString(appointment.status) || "pending",
//       })),
//     };
//   }

//   let recentHealthHistory: AIHealthApplicationContext["recentHealthHistory"] =
//     [];

//   if (intents.history || intents.navigation) {
//     toolsUsed.push("Saved AI health history lookup");

//     const historyDocuments = await database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .find(getAIHealthOwnerFilter(userId))
//       .sort({ updatedAt: -1, createdAt: -1 })
//       .limit(5)
//       .toArray();

//     recentHealthHistory = historyDocuments.map((history) => {
//       const report =
//         typeof history.report === "object" && history.report !== null
//           ? (history.report as Record<string, unknown>)
//           : {};

//       return {
//         id: getDoctorDocumentId(history),
//         title:
//           getDoctorString(history.conversationTitle) ||
//           getDoctorString(report.reportTitle) ||
//           "AI Health Summary",
//         urgencyLevel: getAIHealthUrgency(report.urgencyLevel),
//         updatedAt: formatDoctorDate(history.updatedAt || history.createdAt),
//       };
//     });
//   }

//   if (intents.doctor) {
//     toolsUsed.push("Specialist matching context");
//   }

//   return {
//     user: {
//       id: userId,
//       name: userName,
//       role,
//     },
//     routes,
//     doctorDirectory,
//     appointmentContext,
//     recentHealthHistory,
//     toolsUsed: Array.from(new Set(toolsUsed)),
//     contextMemoryUsed,
//   };
// };

// const buildAIHealthNaturalResponsePrompt = (
//   context: AIHealthApplicationContext,
// ): string => `You are SebaSathi AI Health Assistant, an advanced conversational assistant integrated into a Bangladesh-oriented healthcare application.

// You must do more than simple text generation. Use conversation memory and the supplied SebaSathi application context to answer questions, reason about next steps, help the user navigate the application, and ask useful follow-up questions when information is missing.

// Signed-in user:
// ${JSON.stringify(context.user)}

// SebaSathi application context retrieved by backend tools:
// ${JSON.stringify({
//   routes: context.routes,
//   doctorDirectory: context.doctorDirectory,
//   appointmentContext: context.appointmentContext,
//   recentHealthHistory: context.recentHealthHistory,
//   toolsUsed: context.toolsUsed,
// })}

// Behavior requirements:
// - Answer health questions and SebaSathi application questions naturally.
// - Use previous conversation messages to understand references such as “it”, “that problem”, “same pain”, or “what should I do next”.
// - When application data is available, use it accurately. Never invent appointments, doctors, history, counts, status, dates or routes.
// - If the user asks where to go in the application, explain the correct page and mention the relevant route label naturally.
// - Explain the practical basis for recommendations without revealing hidden chain-of-thought.
// - Ask concise follow-up questions when key details are missing.
// - Match the user's language: easy Bangla, Banglish or English.
// - For health guidance, never confirm a diagnosis, prescribe medicine, provide individualized doses, or advise stopping prescribed treatment.
// - Emergency warning signs require immediate emergency-care advice.
// - Usually write 5-9 clear sentences and approximately 120-240 words when enough information exists.
// - Return only the natural conversational answer. Do not return JSON, markdown tables, internal IDs or hidden reasoning.`;

// const buildAIHealthMetadataPrompt = (
//   context: AIHealthApplicationContext,
//   latestUserMessage: string,
//   assistantReply: string,
// ): string => `Create safe structured metadata for a completed SebaSathi AI assistant reply.

// User message:
// ${latestUserMessage}

// Assistant reply:
// ${assistantReply}

// Allowed navigation routes:
// ${JSON.stringify(context.routes)}

// Backend tools already used:
// ${JSON.stringify(context.toolsUsed)}

// Return ONLY valid JSON with this exact shape:
// {
//   "urgencyLevel": "routine | soon | urgent | emergency",
//   "suggestedSpecialists": ["maximum three specialist categories that exist in or reasonably map to the doctor directory"],
//   "recommendedActions": ["maximum five safe practical actions"],
//   "warningSigns": ["maximum four important warning signs"],
//   "followUpQuestions": ["maximum three useful follow-up questions"],
//   "suggestedPrompts": ["maximum four short prompts the user can click to continue the conversation"],
//   "navigationActions": [
//     {
//       "label": "must correspond to an allowed route",
//       "href": "must exactly match one allowed route href",
//       "reason": "short explanation of why this page is relevant"
//     }
//   ],
//   "decisionBasis": "one or two concise user-facing sentences explaining which reported facts or application context influenced the guidance, without exposing private chain-of-thought",
//   "toolsUsed": ["copy only tools actually listed above"],
//   "contextMemoryUsed": ${context.contextMemoryUsed ? "true" : "false"},
//   "disclaimer": "one short medical disclaimer"
// }

// Do not invent app data. Include navigationActions only when useful. Suggested prompts must be directly usable as the user's next message.`;

// const writeAIHealthStreamEvent = (
//   res: Response,
//   event: Record<string, unknown>,
// ): void => {
//   if (!res.writableEnded) {
//     res.write(`${JSON.stringify(event)}\n`);
//   }
// };

// const startAIHealthStream = (res: Response): void => {
//   res.status(200);
//   res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
//   res.setHeader("Cache-Control", "no-cache, no-transform");
//   res.setHeader("Connection", "keep-alive");
//   res.setHeader("X-Accel-Buffering", "no");
//   res.flushHeaders();
// };

// const writeAIHealthStatus = (
//   res: Response,
//   stage: AIHealthStreamStage,
//   message: string,
//   toolsUsed: string[] = [],
// ): void => {
//   writeAIHealthStreamEvent(res, {
//     type: "status",
//     stage,
//     message,
//     toolsUsed,
//   });
// };

// /* =========================================================
//    AI Health access status
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/access",
//   verifyToken,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           allowed: false,
//           message: "User account was not found",
//         });

//         return;
//       }

//       const role = getNormalizedUserRole(currentUser);
//       const status = getNormalizedUserStatus(currentUser);
//       const allowed = status === "active";

//       res.status(200).json({
//         success: true,
//         authenticated: true,
//         allowed,
//         role,
//         status,
//         user: {
//           id: getDoctorDocumentId(currentUser),
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//         },
//         message: allowed
//           ? "Your active account can use SebaSathi AI Health Assistant."
//           : "Your account is blocked. Contact the administrator to use the AI Health Assistant.",
//       });
//     } catch (error) {
//       console.error("AI Health access error:", error);

//       res.status(500).json({
//         success: false,
//         allowed: false,
//         message: "Failed to verify AI Health access",
//       });
//     }
//   },
// );

// /* =========================================================
//    AI Health conversation history
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/conversations",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const limit = getPositiveInteger(req.query.limit, 100, 100);
//       const conversations = await database
//         .collection(AI_HEALTH_CHAT_COLLECTION)
//         .find(getAIHealthOwnerFilter(userId))
//         .sort({
//           lastMessageAt: -1,
//           updatedAt: -1,
//           _id: -1,
//         })
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         conversations: conversations.map(formatAIHealthConversation),
//       });
//     } catch (error) {
//       console.error("Get AI conversations error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI conversations",
//       });
//     }
//   },
// );

// app.post(
//   "/api/v1/ai-health/conversations",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const now = new Date();
//       const userId = getDoctorDocumentId(currentUser);
//       const userRole = getNormalizedUserRole(currentUser);
//       const userName = getDoctorString(currentUser.name);
//       const userEmail = normalizeDoctorEmail(currentUser.email);
//       const userImage = getDoctorString(currentUser.image) || null;
//       const requestedTitle = getDoctorString(req.body.title).slice(0, 80);

//       const conversationDocument = {
//         title: requestedTitle || "New health chat",
//         userId,
//         userRole,
//         userName,
//         userEmail,
//         userImage,
//         patientUserId: userId,
//         patientName: userName,
//         patientEmail: userEmail,
//         patientImage: userImage,
//         messages: [] as AIHealthStoredMessage[],
//         summaryHistoryId: null,
//         summaryReport: null,
//         createdAt: now,
//         updatedAt: now,
//         lastMessageAt: now,
//       };

//       const collection = database.collection(AI_HEALTH_CHAT_COLLECTION);
//       const insertResult = await collection.insertOne(conversationDocument);
//       const conversation = await collection.findOne({
//         _id: insertResult.insertedId,
//       });

//       res.status(201).json({
//         success: true,
//         message: "New AI health conversation created",
//         conversation: conversation
//           ? formatAIHealthConversation(conversation)
//           : {
//               id: insertResult.insertedId.toHexString(),
//               ...conversationDocument,
//               messageCount: 0,
//             },
//       });
//     } catch (error) {
//       console.error("Create AI conversation error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to create AI conversation",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/ai-health/conversations/:conversationId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         getDoctorDocumentId(currentUser),
//         getDoctorString(req.params.conversationId),
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         conversation: formatAIHealthConversation(conversation),
//       });
//     } catch (error) {
//       console.error("Get AI conversation details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI conversation",
//       });
//     }
//   },
// );

// app.delete(
//   "/api/v1/ai-health/conversations/:conversationId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       await database.collection(AI_HEALTH_CHAT_COLLECTION).deleteOne({
//         _id: conversation._id,
//       });

//       res.status(200).json({
//         success: true,
//         message: "AI health conversation deleted successfully",
//         deletedConversationId: conversationId,
//       });
//     } catch (error) {
//       console.error("Delete AI conversation error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to delete AI conversation",
//       });
//     }
//   },
// );

// /* =========================================================
//    Advanced streamed AI Health message exchange
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/conversations/:conversationId/messages/stream",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     let streamStarted = false;
//     const abortController = new AbortController();

//     res.on("close", () => {
//       if (!res.writableEnded) {
//         abortController.abort();
//       }
//     });

//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const content = getDoctorString(req.body.message);

//       if (!content) {
//         res.status(400).json({
//           success: false,
//           message: "A health or application question is required",
//         });
//         return;
//       }

//       if (content.length > 4000) {
//         res.status(400).json({
//           success: false,
//           message: "A message cannot contain more than 4000 characters",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       const existingMessages = getStoredAIHealthMessages(conversation.messages);
//       const contextMessages: AIHealthMessage[] = [
//         ...existingMessages.map(({ role, content: savedContent }) => ({
//           role,
//           content: savedContent,
//         })),
//         {
//           role: "user",
//           content,
//         },
//       ];

//       const validation = normalizeAIHealthMessages(contextMessages, {
//         requireLatestUser: true,
//         maximumMessages: 26,
//         maximumCharacters: 32000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       startAIHealthStream(res);
//       streamStarted = true;
//       writeAIHealthStatus(
//         res,
//         "thinking",
//         "Understanding your question and previous conversation...",
//       );

//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         content,
//         existingMessages,
//       );

//       writeAIHealthStatus(
//         res,
//         "tool",
//         "Checking relevant SebaSathi context...",
//         applicationContext.toolsUsed,
//       );

//       writeAIHealthStatus(
//         res,
//         "answering",
//         "Preparing a context-aware response...",
//         applicationContext.toolsUsed,
//       );

//       const naturalReply = await callGroqTextStream(
//         [
//           {
//             role: "system",
//             content: buildAIHealthNaturalResponsePrompt(applicationContext),
//           },
//           ...validation.messages,
//         ],
//         (delta) => {
//           writeAIHealthStreamEvent(res, {
//             type: "delta",
//             delta,
//           });
//         },
//         abortController.signal,
//       );

//       writeAIHealthStatus(
//         res,
//         "structuring",
//         "Creating follow-up prompts, navigation actions and decision support...",
//         applicationContext.toolsUsed,
//       );

//       let metadata: Record<string, unknown> = {};

//       try {
//         metadata = await callGroqAI(
//           [
//             {
//               role: "system",
//               content: buildAIHealthMetadataPrompt(
//                 applicationContext,
//                 content,
//                 naturalReply,
//               ),
//             },
//             {
//               role: "user",
//               content: "Return the requested JSON metadata now.",
//             },
//           ],
//           0.1,
//           900,
//         );
//       } catch (metadataError) {
//         console.error(
//           "AI Health metadata generation warning:",
//           metadataError instanceof Error
//             ? metadataError.message
//             : metadataError,
//         );
//       }

//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const assistant = formatAIHealthAssistantResponse(
//         {
//           ...metadata,
//           reply: naturalReply,
//           toolsUsed: applicationContext.toolsUsed,
//           contextMemoryUsed: applicationContext.contextMemoryUsed,
//         },
//         emergencyDetected,
//         applicationContext,
//       );

//       const now = new Date();
//       const userMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "user",
//         content,
//         createdAt: now,
//       };
//       const assistantMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "assistant",
//         content: naturalReply,
//         assistant,
//         createdAt: new Date(),
//       };

//       const nextTitle =
//         existingMessages.some((message) => message.role === "user") ||
//         getDoctorString(conversation.title) !== "New health chat"
//           ? getDoctorString(conversation.title) || "New health chat"
//           : createAIHealthConversationTitle(content);

//       writeAIHealthStatus(
//         res,
//         "saving",
//         "Saving the conversation and memory...",
//         applicationContext.toolsUsed,
//       );

//       const updatedConversation = await database
//         .collection(AI_HEALTH_CHAT_COLLECTION)
//         .findOneAndUpdate(
//           {
//             _id: conversation._id,
//           },
//           {
//             $set: {
//               title: nextTitle,
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: assistantMessage.createdAt,
//               lastMessageAt: assistantMessage.createdAt,
//             },
//             $push: {
//               messages: {
//                 $each: [userMessage, assistantMessage],
//               },
//             },
//           },
//           {
//             returnDocument: "after",
//           },
//         );

//       writeAIHealthStreamEvent(res, {
//         type: "result",
//         data: {
//           success: true,
//           provider: "groq",
//           model: groqModel,
//           userMessage: formatAIHealthConversationMessage(userMessage),
//           assistantMessage: formatAIHealthConversationMessage(assistantMessage),
//           conversation: updatedConversation
//             ? formatAIHealthConversation(updatedConversation)
//             : null,
//         },
//       });

//       res.end();
//     } catch (error) {
//       console.error("AI Health streamed chat error:", error);

//       const message =
//         error instanceof Error
//           ? error.message
//           : "Failed to receive a streamed response from the AI provider";

//       if (streamStarted) {
//         writeAIHealthStreamEvent(res, {
//           type: "error",
//           message,
//         });
//         res.end();
//       } else {
//         res.status(502).json({
//           success: false,
//           message,
//         });
//       }
//     }
//   },
// );

// /* =========================================================
//    Non-streaming persistent message exchange compatibility
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/conversations/:conversationId/messages",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.params.conversationId);
//       const content = getDoctorString(req.body.message);

//       if (!content) {
//         res.status(400).json({
//           success: false,
//           message: "A health or application question is required",
//         });
//         return;
//       }

//       if (content.length > 4000) {
//         res.status(400).json({
//           success: false,
//           message: "A message cannot contain more than 4000 characters",
//         });
//         return;
//       }

//       const conversation = await getAIHealthConversationForUser(
//         userId,
//         conversationId,
//       );

//       if (!conversation) {
//         res.status(404).json({
//           success: false,
//           message: "AI health conversation was not found",
//         });
//         return;
//       }

//       const existingMessages = getStoredAIHealthMessages(conversation.messages);
//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         content,
//         existingMessages,
//       );
//       const contextMessages: AIHealthMessage[] = [
//         ...existingMessages.map(({ role, content: savedContent }) => ({
//           role,
//           content: savedContent,
//         })),
//         {
//           role: "user",
//           content,
//         },
//       ];
//       const validation = normalizeAIHealthMessages(contextMessages, {
//         requireLatestUser: true,
//         maximumMessages: 26,
//         maximumCharacters: 32000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const groqData = await callGroqAI(
//         [
//           {
//             role: "system",
//             content: `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn ONLY JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`,
//           },
//           ...validation.messages,
//         ],
//         0.2,
//         1200,
//       );

//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const assistant = formatAIHealthAssistantResponse(
//         groqData,
//         emergencyDetected,
//         applicationContext,
//       );
//       const now = new Date();
//       const userMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "user",
//         content,
//         createdAt: now,
//       };
//       const assistantMessage: AIHealthStoredMessage = {
//         id: createAIHealthMessageId(),
//         role: "assistant",
//         content: assistant.reply,
//         assistant,
//         createdAt: new Date(),
//       };
//       const nextTitle =
//         existingMessages.some((message) => message.role === "user") ||
//         getDoctorString(conversation.title) !== "New health chat"
//           ? getDoctorString(conversation.title) || "New health chat"
//           : createAIHealthConversationTitle(content);

//       const updatedConversation = await database
//         .collection(AI_HEALTH_CHAT_COLLECTION)
//         .findOneAndUpdate(
//           { _id: conversation._id },
//           {
//             $set: {
//               title: nextTitle,
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: assistantMessage.createdAt,
//               lastMessageAt: assistantMessage.createdAt,
//             },
//             $push: {
//               messages: {
//                 $each: [userMessage, assistantMessage],
//               },
//             },
//           },
//           { returnDocument: "after" },
//         );

//       res.status(200).json({
//         success: true,
//         provider: "groq",
//         model: groqModel,
//         userMessage: formatAIHealthConversationMessage(userMessage),
//         assistantMessage: formatAIHealthConversationMessage(assistantMessage),
//         conversation: updatedConversation
//           ? formatAIHealthConversation(updatedConversation)
//           : null,
//       });
//     } catch (error) {
//       console.error("AI Health persistent chat error:", error);

//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to receive a response from the AI provider",
//       });
//     }
//   },
// );

// /* =========================================================
//    Legacy AI Health chat endpoint
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/chat",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const validation = normalizeAIHealthMessages(req.body.messages, {
//         requireLatestUser: true,
//         maximumMessages: 22,
//         maximumCharacters: 26000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const latestMessage = validation.messages.at(-1)?.content || "";
//       const applicationContext = await buildAIHealthApplicationContext(
//         currentUser,
//         latestMessage,
//         [],
//       );
//       const emergencyDetected = hasEmergencyWarning(validation.messages);
//       const systemPrompt = `${buildAIHealthNaturalResponsePrompt(applicationContext)}\n\nReturn only JSON with reply, urgencyLevel, suggestedSpecialists, recommendedActions, warningSigns, followUpQuestions, suggestedPrompts, navigationActions, decisionBasis, toolsUsed, contextMemoryUsed and disclaimer.`;

//       const groqData = await callGroqAI(
//         [{ role: "system", content: systemPrompt }, ...validation.messages],
//         0.2,
//         1200,
//       );

//       res.status(200).json({
//         success: true,
//         provider: "groq",
//         model: groqModel,
//         assistant: formatAIHealthAssistantResponse(
//           groqData,
//           emergencyDetected,
//           applicationContext,
//         ),
//       });
//     } catch (error) {
//       console.error("AI Health chat error:", error);
//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to receive a response from the AI provider",
//       });
//     }
//   },
// );

// /* =========================================================
//    Generate and save AI Health summary
// ========================================================= */

// app.post(
//   "/api/v1/ai-health/summary",
//   verifyToken,
//   verifyAnyActiveUser,
//   verifyAIHealthRateLimit,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const conversationId = getDoctorString(req.body.conversationId);
//       let conversation: Document | null = null;
//       let conversationTitle = "AI Health Conversation";
//       let messagesValue: unknown = req.body.messages;

//       if (conversationId) {
//         conversation = await getAIHealthConversationForUser(
//           userId,
//           conversationId,
//         );

//         if (!conversation) {
//           res.status(404).json({
//             success: false,
//             message: "AI health conversation was not found",
//           });
//           return;
//         }

//         conversationTitle =
//           getDoctorString(conversation.title) || "AI Health Conversation";
//         messagesValue = getStoredAIHealthMessages(conversation.messages).map(
//           ({ role, content }) => ({ role, content }),
//         );
//       }

//       const validation = normalizeAIHealthMessages(messagesValue, {
//         requireLatestUser: false,
//         maximumMessages: 40,
//         maximumCharacters: 42000,
//       });

//       if (!validation.success) {
//         res.status(400).json({
//           success: false,
//           message: validation.message,
//         });
//         return;
//       }

//       const systemPrompt = `Generate a concise structured health-conversation report for SebaSathi AI. Use only information actually present. Do not invent symptoms, duration, tests, diagnoses or medicines. Do not diagnose or prescribe. Match the user's language where practical.

// Return ONLY valid JSON:
// {
//   "reportTitle": "short title",
//   "conciseSummary": "2-3 concise sentences",
//   "chiefConcerns": ["main concerns"],
//   "symptoms": ["reported symptoms"],
//   "durationAndPattern": "stated duration/pattern or Not clearly stated",
//   "severity": "stated severity or Not clearly stated",
//   "urgencyLevel": "routine | soon | urgent | emergency",
//   "redFlags": ["warning signs"],
//   "suggestedSpecialists": ["specialist categories"],
//   "selfCareGuidance": ["low-risk general guidance"],
//   "questionsForDoctor": ["useful questions"],
//   "emergencyAdvice": "brief emergency advice",
//   "disclaimer": "not a diagnosis or prescription"
// }`;

//       const groqData = await callGroqAI(
//         [
//           {
//             role: "system",
//             content: systemPrompt,
//           },
//           {
//             role: "user",
//             content: JSON.stringify(validation.messages),
//           },
//         ],
//         0.1,
//         1100,
//       );

//       const report = formatAIHealthSummary(groqData);
//       const now = new Date();
//       const userRole = getNormalizedUserRole(currentUser);
//       const userName = getDoctorString(currentUser.name);
//       const userEmail = normalizeDoctorEmail(currentUser.email);
//       const userImage = getDoctorString(currentUser.image) || null;
//       const historyCollection = database.collection(
//         AI_HEALTH_HISTORY_COLLECTION,
//       );

//       const historyDocument = {
//         conversationId: conversation ? getDoctorDocumentId(conversation) : null,
//         conversationTitle,
//         userId,
//         userRole,
//         userName,
//         userEmail,
//         userImage,
//         patientUserId: userId,
//         patientName: userName,
//         patientEmail: userEmail,
//         patientImage: userImage,
//         provider: "groq",
//         model: groqModel,
//         report,
//         messages: validation.messages,
//         createdAt: now,
//         updatedAt: now,
//       };

//       let history: Document | null = null;
//       const existingSummaryId = getDoctorString(conversation?.summaryHistoryId);

//       if (existingSummaryId) {
//         const existingHistory = await historyCollection.findOne({
//           $and: [
//             getDoctorFilter(existingSummaryId),
//             getAIHealthOwnerFilter(userId),
//           ],
//         });

//         if (existingHistory) {
//           history = await historyCollection.findOneAndUpdate(
//             { _id: existingHistory._id },
//             {
//               $set: {
//                 ...historyDocument,
//                 createdAt: existingHistory.createdAt || now,
//                 updatedAt: now,
//               },
//             },
//             { returnDocument: "after" },
//           );
//         }
//       }

//       if (!history) {
//         const insertResult = await historyCollection.insertOne(historyDocument);
//         history = await historyCollection.findOne({
//           _id: insertResult.insertedId,
//         });
//       }

//       if (!history) {
//         throw new Error("The generated summary could not be saved");
//       }

//       if (conversation) {
//         await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
//           { _id: conversation._id },
//           {
//             $set: {
//               summaryHistoryId: getDoctorDocumentId(history),
//               summaryReport: report,
//               updatedAt: now,
//             },
//           },
//         );
//       }

//       res.status(201).json({
//         success: true,
//         message: "AI health summary generated and saved successfully",
//         history: formatAIHealthHistory(history),
//         conversation: conversation
//           ? formatAIHealthConversation({
//               ...conversation,
//               summaryHistoryId: getDoctorDocumentId(history),
//               summaryReport: report,
//               updatedAt: now,
//             })
//           : null,
//       });
//     } catch (error) {
//       console.error("AI Health summary error:", error);

//       res.status(502).json({
//         success: false,
//         message:
//           error instanceof Error
//             ? error.message
//             : "Failed to generate and save the AI health summary",
//       });
//     }
//   },
// );

// /* =========================================================
//    Patient AI Health summary history
//    - Active and blocked patients can read their own history.
//    - Only active patients can delete their own history.
// ========================================================= */

// app.get(
//   "/api/v1/patient/ai-health-history",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const status = getNormalizedUserStatus(currentUser);
//       const requestedPage = getPositiveInteger(req.query.page, 1, 100000);

//       // Patient AI Health History always returns exactly 10 records per page.
//       const limit = 10;
//       const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
//       const filter = getAIHealthOwnerFilter(patientUserId);
//       const total = await collection.countDocuments(filter);
//       const totalPages = Math.max(1, Math.ceil(total / limit));
//       const page = Math.min(requestedPage, totalPages);

//       const documents = await collection
//         .find(filter)
//         .sort({
//           updatedAt: -1,
//           createdAt: -1,
//           _id: -1,
//         })
//         .skip((page - 1) * limit)
//         .limit(limit)
//         .toArray();

//       res.status(200).json({
//         success: true,
//         account: {
//           id: patientUserId,
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//           role: "patient",
//           status,
//         },
//         canDelete: status === "active",
//         histories: documents.map(formatAIHealthHistory),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages,
//         },
//       });
//     } catch (error) {
//       console.error("Get patient AI Health history error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient AI health history",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/patient/ai-health-history/:historyId",
//   verifyToken,
//   verifyPatient,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);

//       if (!historyId) {
//         res.status(400).json({
//           success: false,
//           message: "AI health history ID is required",
//         });
//         return;
//       }

//       const history = await database
//         .collection(AI_HEALTH_HISTORY_COLLECTION)
//         .findOne({
//           $and: [
//             getDoctorFilter(historyId),
//             getAIHealthOwnerFilter(patientUserId),
//           ],
//         });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       const status = getNormalizedUserStatus(currentUser);

//       res.status(200).json({
//         success: true,
//         account: {
//           id: patientUserId,
//           name: getDoctorString(currentUser.name),
//           email: normalizeDoctorEmail(currentUser.email),
//           image: getDoctorString(currentUser.image) || null,
//           role: "patient",
//           status,
//         },
//         canDelete: status === "active",
//         history: formatAIHealthHistory(history),
//       });
//     } catch (error) {
//       console.error("Get patient AI Health history details error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve patient AI health history details",
//       });
//     }
//   },
// );

// app.delete(
//   "/api/v1/patient/ai-health-history/:historyId",
//   verifyToken,
//   verifyPatient,
//   verifyActive,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const patientUserId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);

//       if (!historyId) {
//         res.status(400).json({
//           success: false,
//           message: "AI health history ID is required",
//         });
//         return;
//       }

//       const historyCollection = database.collection(
//         AI_HEALTH_HISTORY_COLLECTION,
//       );

//       const history = await historyCollection.findOne({
//         $and: [
//           getDoctorFilter(historyId),
//           getAIHealthOwnerFilter(patientUserId),
//         ],
//       });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       const deleteResult = await historyCollection.deleteOne({
//         _id: history._id,
//       });

//       if (deleteResult.deletedCount !== 1) {
//         res.status(500).json({
//           success: false,
//           message: "AI health history could not be deleted",
//         });
//         return;
//       }

//       const conversationId = getDoctorString(history.conversationId);

//       if (conversationId) {
//         await database.collection(AI_HEALTH_CHAT_COLLECTION).updateOne(
//           {
//             $and: [
//               getDoctorFilter(conversationId),
//               getAIHealthOwnerFilter(patientUserId),
//               {
//                 summaryHistoryId: historyId,
//               },
//             ],
//           },
//           {
//             $set: {
//               summaryHistoryId: null,
//               summaryReport: null,
//               updatedAt: new Date(),
//             },
//           },
//         );
//       }

//       res.status(200).json({
//         success: true,
//         message: "AI health history deleted successfully",
//         deletedHistoryId: historyId,
//       });
//     } catch (error) {
//       console.error("Delete patient AI Health history error:", error);

//       res.status(500).json({
//         success: false,
//         message: "Failed to delete patient AI health history",
//       });
//     }
//   },
// );

// /* =========================================================
//    Saved AI Health summary history
// ========================================================= */

// app.get(
//   "/api/v1/ai-health/history",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const page = getPositiveInteger(req.query.page, 1, 100000);
//       const limit = 10;
//       const collection = database.collection(AI_HEALTH_HISTORY_COLLECTION);
//       const filter = getAIHealthOwnerFilter(userId);

//       const [documents, total] = await Promise.all([
//         collection
//           .find(filter)
//           .sort({ createdAt: -1, _id: -1 })
//           .skip((page - 1) * limit)
//           .limit(limit)
//           .toArray(),
//         collection.countDocuments(filter),
//       ]);

//       res.status(200).json({
//         success: true,
//         histories: documents.map(formatAIHealthHistory),
//         pagination: {
//           page,
//           limit,
//           total,
//           totalPages: Math.max(1, Math.ceil(total / limit)),
//         },
//       });
//     } catch (error) {
//       console.error("Get AI Health history error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI health history",
//       });
//     }
//   },
// );

// app.get(
//   "/api/v1/ai-health/history/:historyId",
//   verifyToken,
//   verifyAnyActiveUser,
//   async (req: AuthenticatedRequest, res: Response): Promise<void> => {
//     try {
//       if (!database) {
//         res.status(503).json({
//           success: false,
//           message: "Database is not connected",
//         });
//         return;
//       }

//       const currentUser = await getCurrentDatabaseUser(req);

//       if (!currentUser) {
//         res.status(404).json({
//           success: false,
//           message: "User account was not found",
//         });
//         return;
//       }

//       const userId = getDoctorDocumentId(currentUser);
//       const historyId = getDoctorString(req.params.historyId);
//       const history = await database
//         .collection(AI_HEALTH_HISTORY_COLLECTION)
//         .findOne({
//           $and: [getDoctorFilter(historyId), getAIHealthOwnerFilter(userId)],
//         });

//       if (!history) {
//         res.status(404).json({
//           success: false,
//           message: "AI health history was not found",
//         });
//         return;
//       }

//       res.status(200).json({
//         success: true,
//         history: formatAIHealthHistory(history),
//       });
//     } catch (error) {
//       console.error("Get AI Health history details error:", error);
//       res.status(500).json({
//         success: false,
//         message: "Failed to retrieve AI health history details",
//       });
//     }
//   },
// );

// /* =========================================================
//    Unknown route handler
// ========================================================= */

// app.use((_req: Request, res: Response) => {
//   res.status(404).json({
//     success: false,
//     message: "API route not found",
//   });
// });

// /* =========================================================
//    Global error handler
// ========================================================= */

// app.use((error: Error, _req: Request, res: Response, _next: NextFunction) => {
//   console.error("Server error:", error);

//   res.status(500).json({
//     success: false,
//     message: "Internal server error",
//   });
// });

// /* =========================================================
//    MongoDB connection
// ========================================================= */

// const connectDatabase = async (): Promise<void> => {
//   await mongoClient.connect();

//   database = mongoClient.db(mongoDbName);

//   await database.command({ ping: 1 });

//   await Promise.all([
//     database
//       .collection("user")
//       .createIndex({ role: 1, status: 1, updatedAt: -1 }),
//     database.collection("user").createIndex({ role: 1, name: 1 }),
//     database.collection("user").createIndex({ role: 1, email: 1 }),
//     database
//       .collection("doctors")
//       .createIndex({ status: 1, ratingAverage: -1, createdAt: -1 }),
//     database.collection("doctors").createIndex({ name: 1 }),
//     database.collection("doctors").createIndex({ specialization: 1 }),
//     database.collection("doctors").createIndex({ qualification: 1 }),
//     database.collection("doctors").createIndex({ hospital: 1 }),
//     database.collection("doctors").createIndex({ experienceYears: 1 }),
//     database
//       .collection("reviews")
//       .createIndex({ doctorId: 1, userId: 1 }, { unique: true }),
//     database.collection("reviews").createIndex({ doctorId: 1, updatedAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ patientUserId: 1, doctorId: 1, status: 1 }),
//     database
//       .collection("appointments")
//       .createIndex({ patientUserId: 1, createdAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ doctorUserId: 1, status: 1, appointmentDate: 1 }),
//     database
//       .collection("appointments")
//       .createIndex({ doctorUserId: 1, createdAt: -1 }),
//     database
//       .collection("appointments")
//       .createIndex({ status: 1, appointmentDate: 1, appointmentTime: 1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ userId: 1, createdAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ patientUserId: 1, createdAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ userId: 1, updatedAt: -1 }),
//     database
//       .collection(AI_HEALTH_HISTORY_COLLECTION)
//       .createIndex({ patientUserId: 1, updatedAt: -1 }),
//     database
//       .collection(AI_HEALTH_CHAT_COLLECTION)
//       .createIndex({ userId: 1, lastMessageAt: -1 }),
//     database
//       .collection(AI_HEALTH_CHAT_COLLECTION)
//       .createIndex({ patientUserId: 1, lastMessageAt: -1 }),
//   ]);

//   console.log(`MongoDB connected successfully. Database: ${mongoDbName}`);
// };

// /* =========================================================
//    Start server
// ========================================================= */

// const startServer = async (): Promise<void> => {
//   try {
//     await connectDatabase();

//     app.listen(port, () => {
//       console.log(`SebaSathi AI server is running on http://localhost:${port}`);

//       console.log(`JWKS URL: ${jwksUrl.toString()}`);
//     });
//   } catch (error) {
//     console.error(
//       "Unable to start SebaSathi AI server:",
//       error instanceof Error ? error.message : error,
//     );

//     await mongoClient.close();
//     process.exit(1);
//   }
// };

// void startServer();

// /* =========================================================
//    Graceful shutdown
// ========================================================= */

// const shutdownServer = async (signal: string): Promise<void> => {
//   console.log(`${signal} received. Closing MongoDB connection...`);

//   try {
//     await mongoClient.close();

//     console.log("MongoDB connection closed successfully");

//     process.exit(0);
//   } catch (error) {
//     console.error("Error closing MongoDB connection:", error);

//     process.exit(1);
//   }
// };

// process.on("SIGINT", () => {
//   void shutdownServer("SIGINT");
// });

// process.on("SIGTERM", () => {
//   void shutdownServer("SIGTERM");
// });

// export default app;
