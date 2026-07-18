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

    chamber: getDoctorString(doctor.chamber),

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

        chamber: getDoctorString(req.body.chamber),

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

            chamber: getDoctorString(req.body.chamber),

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
