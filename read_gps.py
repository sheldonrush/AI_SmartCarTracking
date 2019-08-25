# Global Variable
import globals

def readGPS(gpsFile):


    file_object1 = open(gpsFile,'r')
    print("gps file: ", gpsFile)

    cnt = 0
    while True:
        line = file_object1.readline()
        if line:
            if (line.find("GPRMC") == 1):

                df = line.split(',')

                # get timestamp (unit : second)
                timestamp = str(df[1])
                hh = int(timestamp[0:2])
                mm = int(timestamp[2:4])
                ss = int(timestamp[4:6])

                if cnt == 0:
                    baseTimeStamp = hh*60+mm*60+ss
                    cnt = 1

                s_ticktime = hh*60+mm*60+ss-baseTimeStamp

                # get speed
                kmh = int(float(df[7]) * 1.852)
                globals.kmhs.append(str(kmh))

                N = float(df[3]) #北緯
                E = float(df[5]) #東經
                Nint = int(N)
                Ngrid1 = int(Nint/100)
                Ngrid2 = int(Nint - Ngrid1 *100)
                Nintdec = N - Nint
                Ngrid3 = int(Nintdec * 100)
#               print("N", df[3])        
#               print('N', Ngrid1,'°', Ngrid2, '\'', Nintdecint, '\"')
#               output: N 24 ° 48 ' 3 "
                globals.Nlist1.append(str(Ngrid1))
                globals.Nlist2.append(str(Ngrid2))    
                globals.Nlist3.append(str(Ngrid3))
        
                Eint = int(E)
                Egrid1 = int(Eint/100)
                Egrid2 = int(Eint - Egrid1 *100)
                Eintdec = E - Eint
                Egrid3 = int(Eintdec * 100)
                globals.Elist1.append(str(Egrid1))
                globals.Elist2.append(str(Egrid2))    
                globals.Elist3.append(str(Egrid3))  

        else:
            break
    file_object1.close()


