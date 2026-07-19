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
    createdAt: formatDoctorDate(appointment.createdAt),
    updatedAt: formatDoctorDate(appointment.updatedAt),
  };
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
   Patient appointments
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

      const appointments = await database
        .collection("appointments")
        .find({
          patientUserId,
        })
        .sort({
          createdAt: -1,
        })
        .toArray();

      res.status(200).json({
        success: true,
        appointments: appointments.map(formatAppointment),
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
  const limit = getPositiveInteger(req.query.limit, 20, 100);

  const queryFilter = getAppointmentListFilter(req);

  const filter: Filter<Document> = {
    $and: [queryFilter, additionalFilter],
  };

  const appointmentsCollection = database.collection("appointments");

  const [appointmentDocuments, total] = await Promise.all([
    appointmentsCollection
      .find(filter)
      .sort({
        appointmentDate: 1,
        appointmentTime: 1,
        createdAt: -1,
      })
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

      if (currentStatus === "completed" || currentStatus === "rejected") {
        res.status(409).json({
          success: false,
          message: "A completed or rejected appointment cannot be changed",
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

      const updatedAppointment = await appointmentsCollection.findOneAndUpdate(
        {
          _id: appointment._id,
        },
        {
          $set: {
            status: requestedStatus as AppointmentStatus,
            updatedAt: new Date(),
          },
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
              ? "Consultation completed successfully. The patient can now take another appointment with this doctor."
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

  await Promise.all([
    database
      .collection("doctors")
      .createIndex({ status: 1, ratingAverage: -1, createdAt: -1 }),
    database.collection("doctors").createIndex({ name: 1 }),
    database.collection("doctors").createIndex({ specialization: 1 }),
    database.collection("doctors").createIndex({ qualification: 1 }),
    database.collection("doctors").createIndex({ hospital: 1 }),
    database.collection("doctors").createIndex({ experienceYears: 1 }),
    database
      .collection("reviews")
      .createIndex({ doctorId: 1, userId: 1 }, { unique: true }),
    database.collection("reviews").createIndex({ doctorId: 1, updatedAt: -1 }),
    database
      .collection("appointments")
      .createIndex({ patientUserId: 1, doctorId: 1, status: 1 }),
    database
      .collection("appointments")
      .createIndex({ doctorUserId: 1, status: 1, appointmentDate: 1 }),
    database
      .collection("appointments")
      .createIndex({ status: 1, appointmentDate: 1, appointmentTime: 1 }),
  ]);

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
